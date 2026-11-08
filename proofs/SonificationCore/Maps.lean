/-
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: 2026 Chethan Dandebala Venkatarama
-/
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# Parameter maps

Definitions mirror the Rust `ParamMap`: with `t = (x − x_lo) / (x_hi − x_lo)`
(`unitPos`), the linear map is `y_lo + t · (y_hi − y_lo)` (`linMap`; with clamping,
`t` is held in [0, 1] first, `clampedLinMap`) and the exponential map is
`f_lo · (f_hi / f_lo)ᵗ` (`logMap`), a log-frequency scale when its outputs are
frequencies.

* `linear_map`: the linear map sends the ends to the ends, is strictly increasing
  when `y_hi > y_lo`, and the clamped map stays between `y_lo` and `y_hi`
  (`sonify:LINEAR-MAP`).
* `log_map`: the exponential map sends the ends to the ends, is strictly
  increasing when `f_hi > f_lo > 0`, and `g(x + d) / g(x)` does not depend on `x`:
  equal steps of the data give equal frequency ratios (`sonify:LOG-MAP`).

These are statements about exact real numbers.
-/

namespace SonificationCore

open Real

/-- Where `x` lies from `xlo` (0) to `xhi` (1): `(x − xlo) / (xhi − xlo)`. -/
noncomputable def unitPos (xlo xhi x : ℝ) : ℝ := (x - xlo) / (xhi - xlo)

/-- Rust `ParamMap::Linear` without clamping: `ylo + t · (yhi − ylo)`. -/
noncomputable def linMap (xlo xhi ylo yhi x : ℝ) : ℝ := ylo + unitPos xlo xhi x * (yhi - ylo)

/-- Rust `ParamMap::Linear` with `clamp`: the position is held in [0, 1] first. -/
noncomputable def clampedLinMap (xlo xhi ylo yhi x : ℝ) : ℝ :=
  ylo + max 0 (min 1 (unitPos xlo xhi x)) * (yhi - ylo)

/-- Rust `ParamMap::Exponential` without clamping: `flo · (fhi / flo)ᵗ`. -/
noncomputable def logMap (xlo xhi flo fhi x : ℝ) : ℝ := flo * (fhi / flo) ^ unitPos xlo xhi x

theorem unitPos_lo (xlo xhi : ℝ) : unitPos xlo xhi xlo = 0 := by simp [unitPos]

theorem unitPos_hi {xlo xhi : ℝ} (h : xlo < xhi) : unitPos xlo xhi xhi = 1 :=
  div_self (sub_pos.mpr h).ne'

theorem unitPos_strictMono {xlo xhi : ℝ} (h : xlo < xhi) : StrictMono (unitPos xlo xhi) :=
  fun _ _ hab => div_lt_div_of_pos_right (by linarith) (sub_pos.mpr h)

/-- **sonify:LINEAR-MAP.** For `xlo < xhi`, the linear map sends `xlo` to `ylo` and `xhi` to
`yhi`, is strictly increasing when `ylo < yhi`, and with clamping its output always lies between
`ylo` and `yhi`. -/
theorem linear_map {xlo xhi : ℝ} (ylo yhi : ℝ) (hx : xlo < xhi) :
    linMap xlo xhi ylo yhi xlo = ylo ∧ linMap xlo xhi ylo yhi xhi = yhi ∧
      (ylo < yhi → StrictMono (linMap xlo xhi ylo yhi)) ∧
      ∀ x, min ylo yhi ≤ clampedLinMap xlo xhi ylo yhi x ∧
        clampedLinMap xlo xhi ylo yhi x ≤ max ylo yhi := by
  refine ⟨by simp [linMap, unitPos_lo], by simp [linMap, unitPos_hi hx], fun hy a b hab => ?_,
    fun x => ?_⟩
  · have := mul_lt_mul_of_pos_right (unitPos_strictMono hx hab) (sub_pos.mpr hy)
    simp only [linMap]
    linarith
  · have h0 : 0 ≤ max 0 (min 1 (unitPos xlo xhi x)) := le_max_left _ _
    have h1 : max 0 (min 1 (unitPos xlo xhi x)) ≤ 1 := max_le zero_le_one (min_le_left _ _)
    simp only [clampedLinMap]
    rcases le_total ylo yhi with h | h
    · rw [min_eq_left h, max_eq_right h]
      constructor <;> nlinarith
    · rw [min_eq_right h, max_eq_left h]
      constructor <;> nlinarith

/-- **sonify:LOG-MAP.** For `xlo < xhi` and `0 < flo < fhi`, the exponential map
`g(x) = flo · (fhi / flo)^((x − xlo) / (xhi − xlo))` sends `xlo` to `flo` and `xhi` to `fhi`,
is strictly increasing, and `g(x + d) / g(x) = (fhi / flo)^(d / (xhi − xlo))` for every `x`:
the ratio over a step `d` does not depend on where the step starts. -/
theorem log_map {xlo xhi flo fhi : ℝ} (hx : xlo < xhi) (h0 : 0 < flo) (h1 : flo < fhi) :
    logMap xlo xhi flo fhi xlo = flo ∧ logMap xlo xhi flo fhi xhi = fhi ∧
      StrictMono (logMap xlo xhi flo fhi) ∧
      ∀ x d, logMap xlo xhi flo fhi (x + d) / logMap xlo xhi flo fhi x =
        (fhi / flo) ^ (d / (xhi - xlo)) := by
  have hr : 1 < fhi / flo := (one_lt_div h0).mpr h1
  have hr0 : 0 < fhi / flo := by linarith
  refine ⟨by simp [logMap, unitPos_lo], ?_, fun a b hab => ?_, fun x d => ?_⟩
  · simp only [logMap, unitPos_hi hx, rpow_one]
    field_simp
  · exact mul_lt_mul_of_pos_left (rpow_lt_rpow_of_exponent_lt hr (unitPos_strictMono hx hab)) h0
  · have hp : 0 < (fhi / flo) ^ unitPos xlo xhi x := rpow_pos_of_pos hr0 _
    have hstep : unitPos xlo xhi (x + d) = unitPos xlo xhi x + d / (xhi - xlo) := by
      simp only [unitPos]
      ring
    simp only [logMap, hstep, rpow_add hr0]
    field_simp

end SonificationCore
