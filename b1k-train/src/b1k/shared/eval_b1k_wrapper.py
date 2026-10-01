"""B1K policy wrapper with action compression, rolling inpainting, and stage voting."""

import logging
import os
import numpy as np
import torch
import dataclasses
from collections import deque

from openpi_client.base_policy import BasePolicy
from openpi_client.image_tools import resize_with_pad
from b1k.policies.b1k_policy import extract_state_from_proprio
from b1k.models.pi_behavior_config import BDDL_TASK_NUM_STAGES, TASK_NUM_STAGES
from b1k.shared.correction_rules import apply_correction_rules, check_gripper_variation
from b1k.shared.proprio import PROPRIOCEPTION_INDICES  # 2026 R1Pro layout, vendored (no OmniGibson needed)

logger = logging.getLogger(__name__)

RESIZE_SIZE = 224

# Diagnostic: feed the model a deliberately wrong stage while the voting / correction logic keeps
# tracking the real one. B1K_STAGE_OVERRIDE = "fixed:<n>" (model always sees stage n) or
# "shift:<k>" (model sees tracked stage + k, clamped). Unset = normal behaviour.
STAGE_OVERRIDE = os.environ.get("B1K_STAGE_OVERRIDE", "").strip() or None

# Which head drives the tracked stage, overriding what the checkpoint's config implies.
# B1K_EVENT_COUNTER = "1" (event head -> rising-edge counter) or "0" (15-way head -> voting tracker).
# Unset follows the model config.
#
# A checkpoint trained with use_bddl_event carries both heads, and both are logged on every vote, so
# the same checkpoint can be evaluated either way without retraining. Note this only chooses which
# readout *drives* the robot: the history observations are part of the trained prefix and are always
# supplied, whichever readout is selected.
_EVENT_COUNTER_ENV = os.environ.get("B1K_EVENT_COUNTER", "").strip()
EVENT_COUNTER_OVERRIDE = None if _EVENT_COUNTER_ENV == "" else _EVENT_COUNTER_ENV not in ("0", "false", "False")


@dataclasses.dataclass
class B1KWrapperConfig:
    """Configuration for B1K policy wrapper execution parameters."""
    actions_to_execute: int = 26
    actions_to_keep: int = 4
    execute_in_n_steps: int = 20
    history_len: int = 3
    votes_to_promote: int = 2
    time_threshold_inpaint: float = 0.3
    num_steps: int = 20
    apply_eval_tricks: bool = True
    # Must match the served checkpoint's model config. With use_bddl_stage the model is conditioned
    # on the BDDL symbolic progress (how many goal literals hold) rather than on the time-split
    # stage: tokenized_prompt gains a third slot for it, the time-split slot is pinned to 0 exactly
    # as in training, and the VLM's stage head -- whose logits drive the voting below -- predicts
    # the symbolic stage, so the per-task stage counts come from BDDL_TASK_NUM_STAGES.
    use_bddl_stage: bool = False
    # Never condition the policy on the terminal stage. The evaluator ends an episode the instant the
    # goal really holds, so while an episode is still running the task is by definition unfinished and
    # a terminal-stage reading can only be a false positive -- and acting on it tells the policy it is
    # done, so it stops. In the 2026-09-26 run 5 of the 6 failures sat at the terminal stage for
    # 49-73% of the episode (instance 304 with nothing in the bin at all), burning 3862-5742 steps.
    # Only applied where there is a stage left to fall back to (>= 3 stages); with 2 stages the clamp
    # would pin the signal to 0, which the fixed0 ablation showed destroys the policy.
    cap_terminal_stage: bool = True

    # Frames back for the second observation, or 0 for none. Must match the served checkpoint's
    # PiBehaviorConfig.history_frames.
    #
    # No extra simulator access is needed: the policy is consulted every execute_in_n_steps steps, so
    # the observation from history_frames // execute_in_n_steps calls ago *is* the frame we want. With
    # the defaults that is exactly two calls back (40 = 2 x 20).
    history_frames: int = 0

    # Drive the stage from the event head instead of the voting tracker.
    #
    # The tracker promotes on votes_to_promote of history_len and may skip two stages at once, while
    # demotion needs unanimity and moves one stage -- so a fluctuating input ratchets upward. Measured
    # on sorting_vegetables: the 15-way head's answer changes 178 times per episode where the true
    # stage changes 4.6 times, and the tracker ended +3.9 stages above the truth on average. No setting
    # of history_len / votes_to_promote beat simply emitting the constant 4 (best MAE 3.40 vs 2.00),
    # and prediction confidence does not separate right from wrong (median margin 4.50 vs 5.25), so
    # the input cannot be filtered into shape either.
    use_event_counter: bool = False

    # Inference calls to ignore after counting an event.
    #
    # Two jobs. The 40-frame window overlaps consecutive calls 20 steps apart, so one real event makes
    # about two calls in a row answer "yes" -- rising-edge detection alone would double count on the
    # boundary. And it puts a ceiling on how fast the stage can climb, which the voting tracker lacks.
    # Real stages last 861 frames on average, about 43 calls, so 12 misses nothing genuine.
    event_refractory: int = 12

    # Decision threshold on the event logit. 0.0 is p = 0.5; raise it to demand more confidence.
    event_threshold: float = 0.0


class B1KPolicyWrapper():
    """B1K policy wrapper for PI_BEHAVIOR models with action compression, rolling inpainting, and stage voting."""
    
    def __init__(
        self, 
        policy: BasePolicy,
        text_prompt: str = "PI_BEHAVIOR model (task-conditioned)",  # Not used, kept for compatibility
        action_horizon: int = 30,
        task_id: int | None = None,
        config: B1KWrapperConfig = None,
        checkpoint_switcher = None,
    ) -> None:
        self.base_policy = policy
        self.policy = policy
        self.checkpoint_switcher = checkpoint_switcher
        self.text_prompt = text_prompt
        self.action_horizon = action_horizon
        self.config = config if config is not None else B1KWrapperConfig()
        
        # Validate configuration
        if self.config.actions_to_execute + self.config.actions_to_keep > self.action_horizon:
            raise ValueError(
                f"actions_to_execute + actions_to_keep exceeds action_horizon"
            )
        
        # PI_BEHAVIOR specific (always True for B1K)
        self.task_id = task_id
        self.current_stage = 0
        self.prediction_history = deque([], maxlen=self.config.history_len)
        
        # Control loop variables
        self.last_actions = None
        self.action_index = 0
        self.step_count = 0
        self.prediction_count = 0
        self.next_initial_actions = None

        # History observations, one entry per inference call. Depth is how many calls back the wanted
        # frame is; +1 so that after appending the present the oldest entry is exactly that far back.
        self.history_depth = (
            max(1, round(self.config.history_frames / self.config.execute_in_n_steps))
            if self.config.history_frames > 0 else 0
        )
        self.obs_history = deque([], maxlen=self.history_depth + 1)

        # Event counter state
        self.last_event_logit = None   # read by update_current_stage, logged by serve_ilia_logged
        self.event_prev_fired = False
        self.event_refractory_left = 0
        self.event_count = 0

        if EVENT_COUNTER_OVERRIDE is not None and EVENT_COUNTER_OVERRIDE != self.config.use_event_counter:
            logger.info(
                f"🧪 B1K_EVENT_COUNTER={_EVENT_COUNTER_ENV!r}: stage driven by "
                f"{'the event head' if EVENT_COUNTER_OVERRIDE else 'the 15-way head (voting tracker)'}, "
                f"overriding the checkpoint's config ({self.config.use_event_counter})"
            )
            self.config = dataclasses.replace(self.config, use_event_counter=EVENT_COUNTER_OVERRIDE)

        if self.config.use_event_counter and self.config.history_frames <= 0:
            raise ValueError("use_event_counter needs history_frames > 0 (the event head needs two frames)")
        if self.config.history_frames > 0:
            logger.info(
                f"History observations: {self.config.history_frames} frames back = "
                f"{self.history_depth} inference call(s) at {self.config.execute_in_n_steps} steps each"
            )
        if self.config.use_event_counter:
            logger.info(
                f"Stage from event counter (refractory {self.config.event_refractory} calls, "
                f"threshold {self.config.event_threshold}), voting tracker disabled"
            )

    def _reset_history_and_event_state(self):
        self.obs_history.clear()
        self.last_event_logit = None
        self.event_prev_fired = False
        self.event_refractory_left = 0
        self.event_count = 0


    def num_stages(self, task_id: int) -> int:
        """Stages this task has, under whichever stage definition the served model was trained on."""
        counts = BDDL_TASK_NUM_STAGES if self.config.use_bddl_stage else TASK_NUM_STAGES
        return counts[task_id]

    def reset(self):
        """Reset policy state."""
        self.policy.reset()
        self.last_actions = None
        self.action_index = 0
        self.step_count = 0
        self.prediction_count = 0
        self.next_initial_actions = None
        self.current_stage = 0
        self.prediction_history.clear()
        self._reset_history_and_event_state()
        logger.info(f"Policy reset - Task ID: {self.task_id}, Action horizon: {self.action_horizon}")
    
    def _handle_task_change(self, new_task_id):
        """Handle task ID change by switching checkpoint and resetting state."""
        if self.task_id != new_task_id:
            old_task_id = self.task_id
            self.task_id = new_task_id
            
            logger.info(f"🔄 Task change detected: {old_task_id} → {new_task_id} (max stages: {self.num_stages(new_task_id)})")
            
            if self.checkpoint_switcher:
                new_policy = self.checkpoint_switcher.get_policy_for_task(new_task_id)
                if new_policy is not self.policy:
                    logger.info(f"📦 Switching checkpoint: task {old_task_id} → {new_task_id}")
                    self.base_policy = new_policy
                    self.policy = new_policy
                    self.policy.reset()
            
            self.current_stage = 0
            self.prediction_history.clear()
            self.last_actions = None
            self.action_index = 0
            self.next_initial_actions = None
            self._reset_history_and_event_state()

    def process_obs(self, obs: dict) -> dict:
        """Process observation to match model input format."""
        prop_state = obs["robot_r1::proprio"]
        
        head_original = obs["robot_r1::robot_r1:zed_link:Camera:0::rgb"][..., :3]
        left_original = obs["robot_r1::robot_r1:left_realsense_link:Camera:0::rgb"][..., :3]
        right_original = obs["robot_r1::robot_r1:right_realsense_link:Camera:0::rgb"][..., :3]
        
        # Resize images
        head_resized = resize_with_pad(head_original, RESIZE_SIZE, RESIZE_SIZE)
        left_resized = resize_with_pad(left_original, RESIZE_SIZE, RESIZE_SIZE)
        right_resized = resize_with_pad(right_original, RESIZE_SIZE, RESIZE_SIZE)
        
        cameras = {
            "observation/egocentric_camera": head_resized,
            "observation/wrist_image_left": left_resized,
            "observation/wrist_image_right": right_resized,
        }

        if self.config.history_frames > 0:
            cameras = self._attach_history(cameras)

        return {
            **cameras,
            "observation/state": prop_state,
            "prompt": self.text_prompt,
        }

    def _attach_history(self, cameras: dict) -> dict:
        """Stack each camera as ``[past, present]``, the shape the training loader produces.

        Going through the same two-frame stack that ``delta_timestamps`` yields means B1kInputs splits
        it with the same code in both places, so there is no second definition of which frame is which.

        Until the buffer has filled, the oldest entry available is used -- at the first call that is the
        present frame itself. That mirrors training, where the loader clamps the offset to frame 0 at
        the start of an episode: history equals the present, so the event label reads 0 and the head is
        asked the same question it was trained on rather than something it has never seen.
        """
        self.obs_history.append({k: v.copy() for k, v in cameras.items()})
        past = self.obs_history[0]
        return {k: np.stack([past[k], v], axis=0) for k, v in cameras.items()}


    def update_current_stage(self, predicted_subtask_logits):
        """Advance the tracked stage.

        Two implementations behind config.use_event_counter. Both are entered here rather than from
        act() so that the stage log's patch -- which wraps this method -- records every stage movement
        whichever one is active, and so the 15-way logits keep being logged either way. That is what
        makes the two readouts comparable offline against ``q_score x num_stages``, which is the exact
        final stage of any rollout.
        """
        if self.task_id is None:
            return

        if self.config.use_event_counter:
            self.prediction_history.append(int(np.argmax(predicted_subtask_logits)))
            return self._update_stage_from_event()

        max_stage = self.num_stages(self.task_id) - 1
        predicted_stage = int(np.argmax(predicted_subtask_logits))
        
        if predicted_stage > max_stage:
            predicted_stage = max_stage
        
        self.prediction_history.append(predicted_stage)
        
        if len(self.prediction_history) == self.config.history_len:
            next_stage = self.current_stage + 1
            
            if next_stage <= max_stage:
                votes_for_next = sum(1 for pred in self.prediction_history if pred == next_stage)
                votes_to_skip = sum(1 for pred in self.prediction_history if pred == next_stage + 1)
                votes_to_go_back = sum(1 for pred in self.prediction_history if pred == self.current_stage - 1)
                
                if votes_for_next >= self.config.votes_to_promote:
                    old_stage = self.current_stage
                    self.current_stage = next_stage
                    self.prediction_history.clear()
                    logger.info(f"⬆️  Stage advanced: {old_stage} → {self.current_stage} (task {self.task_id}, step {self.step_count})")
                elif votes_to_skip == self.config.history_len:
                    old_stage = self.current_stage
                    self.current_stage = next_stage
                    self.prediction_history.clear()
                    logger.info(f"⏭️  Stage skipped: {old_stage} → {self.current_stage} (task {self.task_id}, step {self.step_count})")
                elif votes_to_go_back == self.config.history_len and self.current_stage > 0:
                    old_stage = self.current_stage
                    self.current_stage -= 1
                    self.prediction_history.clear()
                    logger.info(f"⬅️  Stage went back: {old_stage} → {self.current_stage} (task {self.task_id}, step {self.step_count})")

    def _update_stage_from_event(self):
        """Stage as a count of detected events: rising edge, then a refractory pause.

        Counting is valid because the BDDL labels are all but monotone -- across 200 demonstrations per
        task only 1-2 episodes contain a single decrement -- and every transition is +1. So the stage is
        simply how many events have happened, and no second head is needed to say where to jump to.

        The two guards do different work. The rising edge stops one event being counted twice while its
        window still overlaps the next call. The refractory bounds the climb rate at something
        physically possible, which is the property the voting tracker never had.
        """
        if self.last_event_logit is None:
            return

        fired = float(self.last_event_logit) > self.config.event_threshold

        if self.event_refractory_left > 0:
            self.event_refractory_left -= 1
            # Still tracked, so an event that keeps firing through the pause is not counted again the
            # moment the pause ends.
            self.event_prev_fired = fired
            return

        rising = fired and not self.event_prev_fired
        self.event_prev_fired = fired
        if not rising:
            return

        max_stage = self.num_stages(self.task_id) - 1
        if self.current_stage >= max_stage:
            return

        old_stage = self.current_stage
        self.current_stage += 1
        self.event_count += 1
        self.event_refractory_left = self.config.event_refractory
        self.prediction_history.clear()
        logger.info(
            f"🔔 Event #{self.event_count}: stage {old_stage} → {self.current_stage} "
            f"(logit {float(self.last_event_logit):+.3f}, task {self.task_id}, step {self.step_count})"
        )

    def _stage_for_model(self) -> int:
        """The stage the model is conditioned on.

        Clamped below the terminal stage (see B1KWrapperConfig.cap_terminal_stage); the tracked stage
        itself is left alone so the logs still show what the VLM believed. B1K_STAGE_OVERRIDE bypasses
        the clamp -- it exists to feed a deliberately wrong stage, so it must not be second-guessed.
        """
        if self.task_id is None:
            return self.current_stage
        max_stage = self.num_stages(self.task_id) - 1
        if STAGE_OVERRIDE is None:
            cap = max_stage - 1 if (self.config.cap_terminal_stage and max_stage >= 2) else max_stage
            stage = min(self.current_stage, cap)
            if stage != self.current_stage and self.prediction_count % 25 == 0:
                logger.info(
                    f"⛔ stage clamped {self.current_stage} -> {stage} (terminal stage cannot hold "
                    f"while the episode is still running; task {self.task_id}, step {self.step_count})"
                )
            return stage
        mode, value = STAGE_OVERRIDE.split(":")
        if mode == "fixed":
            stage = int(value)
        elif mode == "shift":
            stage = self.current_stage + int(value)
        else:
            raise ValueError(f"B1K_STAGE_OVERRIDE={STAGE_OVERRIDE!r}: expected fixed:<n> or shift:<k>")
        return max(0, min(stage, max_stage))

    def prepare_batch_for_pi_behavior(self, batch):
        """Prepare batch for PI_BEHAVIOR model by adding task_id and current_stage."""
        task_id = self.task_id if self.task_id is not None else -1
        batch_copy = batch.copy()
        if "prompt" in batch_copy:
            del batch_copy["prompt"]
        
        stage = self._stage_for_model()
        if STAGE_OVERRIDE is not None and self.prediction_count % 10 == 0:
            logger.info(f"🧪 Stage override {STAGE_OVERRIDE}: model sees stage {stage}, tracked stage {self.current_stage}")
        if self.config.use_bddl_stage:
            # [task_id, time-split stage (pinned to 0, as in training), BDDL symbolic stage]
            entries = [task_id, 0, stage]
        else:
            entries = [task_id, stage]
        batch_copy["tokenized_prompt"] = np.array(entries, dtype=np.int32)
        batch_copy["tokenized_prompt_mask"] = np.ones(len(entries), dtype=bool)
        batch_copy["subtask_state"] = np.array(stage, dtype=np.int32)
        
        return batch_copy
    
    def _interpolate_actions(self, actions, target_steps):
        """Interpolate actions using cubic spline."""
        from scipy.interpolate import interp1d
        
        original_indices = np.linspace(0, len(actions)-1, len(actions))
        target_indices = np.linspace(0, len(actions)-1, target_steps)
        
        interpolated = np.zeros((target_steps, actions.shape[1]))
        for dim in range(actions.shape[1]):
            f = interp1d(original_indices, actions[:, dim], kind='cubic')
            interpolated[:, dim] = f(target_indices)
        
        return interpolated

    def act(self, obs: dict) -> torch.Tensor:
        """Main action function."""
        
        # Extract task_id from observations
        if "task_id" in obs:
            new_task_id = int(obs["task_id"][0])
            self._handle_task_change(new_task_id)
        
        raw_state = obs["robot_r1::proprio"]
        current_state = extract_state_from_proprio(raw_state)
        
        # Check if we need new actions
        if self.last_actions is None or self.action_index >= self.config.execute_in_n_steps:
            
            # Process observation
            model_input = self.process_obs(obs)
            model_input = self.prepare_batch_for_pi_behavior(model_input)
            
            # Add rolling inpainting if available
            if self.next_initial_actions is not None and ("initial_actions" not in model_input or model_input["initial_actions"] is None):
                model_input["initial_actions"] = self.next_initial_actions
            
            # Get prediction
            if "initial_actions" in model_input and model_input["initial_actions"] is not None:
                output = self.policy.infer(model_input, initial_actions=model_input["initial_actions"])
            else:
                output = self.policy.infer(model_input)
            
            actions = output["actions"]
            
            # Ensure correct shape
            if len(actions.shape) == 3:
                actions = actions[0]
            if actions.shape[1] > 23:
                actions = actions[:, :23]
            
            # Apply eval tricks if enabled
            should_compress = self.config.execute_in_n_steps < self.config.actions_to_execute
            
            if self.config.apply_eval_tricks:
                if self.task_id is not None:
                    actions_before = actions.copy()
                    actions, corrected_stage = apply_correction_rules(
                        self.task_id, self.current_stage, current_state, actions
                    )
                    
                    # Log if stage was corrected
                    if corrected_stage != self.current_stage:
                        logger.info(f"🔧 Correction rule: Stage corrected {self.current_stage} → {corrected_stage} (task {self.task_id}, step {self.step_count})")
                        self.current_stage = corrected_stage
                        self.prediction_history.clear()
                    
                    # Log if actions were modified
                    if not np.allclose(actions_before, actions, rtol=1e-3):
                        max_diff = np.max(np.abs(actions_before - actions))
                        logger.info(f"🔧 Correction rule: Actions modified (max diff: {max_diff:.4f}, task {self.task_id}, stage {self.current_stage})")
                
                if should_compress:
                    has_high_variation, mean_var, max_var = check_gripper_variation(
                        actions, self.config.actions_to_execute
                    )
                    if has_high_variation:
                        should_compress = False
                        logger.info(f"🔧 Gripper variation: Compression disabled (mean: {mean_var:.4f}, max: {max_var:.4f})")
            
            # Determine execution parameters
            actions_to_execute = self.config.actions_to_execute if should_compress else self.config.execute_in_n_steps
            execute_steps = self.config.execute_in_n_steps
            
            # Save actions for next inpainting (before compression)
            inpainting_start = actions_to_execute
            inpainting_end = inpainting_start + self.config.actions_to_keep
            
            if len(actions) >= inpainting_end:
                self.next_initial_actions = actions[inpainting_start:inpainting_end].copy()
            else:
                self.next_initial_actions = None
            
            # Extract and compress actions
            self.last_actions = actions[:actions_to_execute].copy()
            
            if should_compress:
                compressed_actions = self._interpolate_actions(self.last_actions, execute_steps)
                compression_factor = actions_to_execute / execute_steps
                compressed_actions[:, :3] *= compression_factor  # Scale velocities
                self.last_actions = compressed_actions
            
            self.action_index = 0
            self.prediction_count += 1
            
            # Log prediction details (at lower frequency, every 10 predictions)
            if self.prediction_count % 10 == 0:
                compression_status = f"compressed {actions_to_execute}→{execute_steps}" if should_compress else f"uncompressed ({execute_steps})"
                logger.info(f"🎯 Prediction #{self.prediction_count} | Actions: {compression_status} | Inpainting: {self.next_initial_actions is not None}")
            
            # Stash the event logit before the stage update: _update_stage_from_event reads it from
            # here rather than taking it as an argument, so update_current_stage keeps the one-argument
            # signature that serve_ilia_logged.py's logging patch wraps.
            if "event_logit" in output:
                self.last_event_logit = float(np.asarray(output["event_logit"]).reshape(-1)[0])

            # Update stage based on model predictions
            if "subtask_logits" in output:
                self.update_current_stage(output["subtask_logits"])
        
        # Get current action from sequence
        if self.action_index >= len(self.last_actions):
            self.action_index = 0
            
        current_action = self.last_actions[self.action_index]
        self.action_index += 1
        self.step_count += 1
        
        # Log progress every 100 steps
        if self.step_count % 100 == 0:
            logger.info(f"📊 Step {self.step_count} | Task: {self.task_id} | Stage: {self.current_stage}/{self.num_stages(self.task_id)-1} | Predictions: {self.prediction_count}")
        
        # Convert to torch tensor
        action_tensor = torch.from_numpy(current_action).float()
        if len(action_tensor) > 23:
            action_tensor = action_tensor[:23]
        
        return action_tensor

