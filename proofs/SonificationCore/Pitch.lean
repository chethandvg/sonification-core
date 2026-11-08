/-
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: 2026 Chethan Dandebala Venkatarama
-/
import PitchMath.Basic
import PitchMath.Octaves

/-!
# Octave shifts

The Rust pitch map `rhythm` moves a frequency by `n` octaves with pitch-math's
`octave_transpose`, `f ↦ f · 2ⁿ` (`PitchMath.octaveTranspose`, which is
rhythm-units' `scaleFrequency f (2ⁿ)`). pitch-math already proves that this keeps
every frequency ratio (`PitchMath.octave_transpose_ratios`,
`pitch:OCTAVE-TRANSPOSE-RATIOS`); `octave_shift` reuses that proof and adds that
every interval in cents is kept too (`sonify:OCTAVE-SHIFT`).
-/

namespace SonificationCore

open PitchMath RhythmUnits

/-- **sonify:OCTAVE-SHIFT.** Moving two frequencies by the same `n` octaves keeps their ratio
(pitch-math's proof) and the interval between them in cents. -/
theorem octave_shift (f₁ f₂ : ℝ) (n : ℤ) :
    ratio (octaveTranspose f₁ n) (octaveTranspose f₂ n) = ratio f₁ f₂ ∧
      cents (octaveTranspose f₁ n) (octaveTranspose f₂ n) = cents f₁ f₂ := by
  refine ⟨octave_transpose_ratios f₁ f₂ n, ?_⟩
  have h := octave_transpose_ratios f₂ f₁ n
  simp only [ratio] at h
  simp only [cents, h]

end SonificationCore
