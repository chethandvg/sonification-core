/-
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: 2026 Chethan Dandebala Venkatarama
-/
import RhythmUnits.Basic

/-!
# The time map

Definitions mirror the Rust functions of the same names. Real time `t` is heard
at audio time `offset + t / k` for a speed-up `k > 0` (`timeMap`, Rust
`TimeMap::to_audio`): rhythm-units' `scalePeriod t k` moved by `offset`. The map
is strictly increasing, keeps every ratio of durations, and `timeInv` (Rust
`TimeMap::to_real`) undoes it (`time_map`, `sonify:TIME-MAP`).

These are statements about exact real numbers; the Rust code's floating-point
rounding is checked by the golden vectors, not here.
-/

namespace SonificationCore

open RhythmUnits

/-- Rust `TimeMap::to_audio`: the audio time `offset + t / k` of real time `t`. -/
noncomputable def timeMap (offset k t : ℝ) : ℝ := offset + scalePeriod t k

/-- Rust `TimeMap::to_real`: the real time `(a − offset) · k` of audio time `a`. -/
def timeInv (offset k a : ℝ) : ℝ := (a - offset) * k

/-- A span of real time lasts `1 / k` as long in the audio. -/
theorem timeMap_sub (offset k t₁ t₂ : ℝ) :
    timeMap offset k t₂ - timeMap offset k t₁ = (t₂ - t₁) / k := by
  simp only [timeMap, scalePeriod]
  ring

theorem timeMap_strictMono {offset k : ℝ} (hk : 0 < k) : StrictMono (timeMap offset k) := by
  intro a b hab
  have h := timeMap_sub offset k a b
  have : 0 < (b - a) / k := div_pos (sub_pos.mpr hab) hk
  linarith

theorem timeMap_ratio (offset : ℝ) {k : ℝ} (hk : k ≠ 0) (t₁ t₂ t₃ : ℝ) :
    (timeMap offset k t₃ - timeMap offset k t₂) / (timeMap offset k t₂ - timeMap offset k t₁) =
      (t₃ - t₂) / (t₂ - t₁) := by
  rw [timeMap_sub, timeMap_sub, div_div_div_cancel_right₀ hk]

theorem timeInv_timeMap (offset : ℝ) {k : ℝ} (hk : k ≠ 0) (t : ℝ) :
    timeInv offset k (timeMap offset k t) = t := by
  simp only [timeInv, timeMap, scalePeriod]
  field_simp
  ring

theorem timeMap_timeInv (offset : ℝ) {k : ℝ} (hk : k ≠ 0) (a : ℝ) :
    timeMap offset k (timeInv offset k a) = a := by
  simp only [timeInv, timeMap, scalePeriod]
  field_simp
  ring

/-- **sonify:TIME-MAP.** For a speed-up `k > 0`, the time map `t ↦ offset + t / k` is strictly
increasing, keeps every ratio of durations,
`(a(t₃) − a(t₂)) / (a(t₂) − a(t₁)) = (t₃ − t₂) / (t₂ − t₁)`, and `timeInv` undoes it both ways. -/
theorem time_map (offset : ℝ) {k : ℝ} (hk : 0 < k) :
    StrictMono (timeMap offset k) ∧
      (∀ t₁ t₂ t₃ : ℝ,
        (timeMap offset k t₃ - timeMap offset k t₂) / (timeMap offset k t₂ - timeMap offset k t₁) =
          (t₃ - t₂) / (t₂ - t₁)) ∧
      (∀ t, timeInv offset k (timeMap offset k t) = t) ∧
      ∀ a, timeMap offset k (timeInv offset k a) = a :=
  ⟨timeMap_strictMono hk, timeMap_ratio offset hk.ne', timeInv_timeMap offset hk.ne',
    timeMap_timeInv offset hk.ne'⟩

end SonificationCore
