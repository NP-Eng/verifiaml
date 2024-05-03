use hcs_common::{quantise_f32_u8_nne, Model, Poly, Tensor};

use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly_commit::PolynomialCommitment;

// Auxiliary function
fn unpadded_inference<F, S, PCS>(
    raw_input: Tensor<f32>,
    model: &Model<i8>,
    qinfo: (f32, u8),
) -> Tensor<u8>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    let quantised_input: Tensor<u8> = Tensor::new(
        quantise_f32_u8_nne(raw_input.values(), qinfo.0, qinfo.1),
        raw_input.shape().clone(),
    );

    let input_i8 = (quantised_input.cast::<i32>() - 128).cast::<i8>();

    let output_i8 = model.evaluate(input_i8);

    (output_i8.cast::<i32>() + 128).cast()
}

// TODO: once padded_evaluate has moved to a common trait, re-insert this
// If padded inference is left on the prover side, move this to the prover
/* // Auxiliary function
fn padded_inference<F, S, PCS>(
    raw_input: Tensor<f32>,
    model: &Model<i8>,
    qinfo: (f32, u8),
) -> Tensor<u8>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    let quantised_input: Tensor<u8> = Tensor::new(
        quantise_f32_u8_nne(raw_input.values(), qinfo.0, qinfo.1),
        raw_input.shape().clone(),
    );

    let input_i8 = (quantised_input.cast::<i32>() - 128).cast::<i8>();

    let output_i8 =
        <Model<i8> as ProveModel<F, S, PCS, i8>>::padded_evaluate(model, input_i8);

    (output_i8.cast::<i32>() + 128).cast()
} */

pub fn run_unpadded<F, S, PCS>(
    input_path: &str,
    expected_output_path: &str,
    model: &Model<i8>,
    qinfo: (f32, u8),
) where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    let raw_input: Tensor<f32> = Tensor::read(input_path);
    let expected_output: Tensor<u8> = Tensor::read(expected_output_path);

    let output_u8 = unpadded_inference::<F, S, PCS>(raw_input, model, qinfo);

    assert_eq!(output_u8, expected_output);

    println!("Single unpadded compatibility test successful");
}

// TODO: once padded_evaluate has moved to a common trait, re-insert this
// If padded inference is left on the prover side, move this to the prover
/* pub fn run_padded<F, S, PCS>(
    input_path: &str,
    expected_output_path: &str,
    model: &Model<i8>,
    qinfo: (f32, u8),
) where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    let raw_input: Tensor<f32> = Tensor::read(input_path);
    let expected_output: Tensor<u8> = Tensor::read(expected_output_path);

    let output_u8 = padded_inference::<F, S, PCS>(raw_input, model, qinfo);

    assert_eq!(output_u8, expected_output);

    println!("Single padded compatibility test successful");
} */

pub fn multi_run_unpadded<F, S, PCS>(
    inputs_path: &str,
    expected_outputs_path: &str,
    model: &Model<i8>,
    qinfo: (f32, u8),
) where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    let raw_inputs: Vec<Tensor<f32>> = Tensor::read_list(inputs_path);
    let expected_outputs: Vec<Tensor<u8>> = Tensor::read_list(expected_outputs_path);

    for (raw_input, expected_output) in raw_inputs.into_iter().zip(expected_outputs.into_iter()) {
        assert_eq!(
            unpadded_inference::<F, S, PCS>(raw_input, model, qinfo),
            expected_output
        );
    }

    println!("Multiple unpadded compatibility test successful");
}

// TODO: once padded_evaluate has moved to a common trait, re-insert this
// If padded inference is left on the prover side, move this to the prover
/*
pub fn multi_run_padded<F, S, PCS>(
    inputs_path: &str,
    expected_outputs_path: &str,
    model: &Model<i8, i32>,
    qinfo: (f32, u8),
) where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    let raw_inputs: Vec<Tensor<f32>> = Tensor::read_list(inputs_path);
    let expected_outputs: Vec<Tensor<u8>> = Tensor::read_list(expected_outputs_path);

    for (raw_input, expected_output) in raw_inputs.into_iter().zip(expected_outputs.into_iter()) {
        assert_eq!(
            padded_inference::<F, S, PCS>(raw_input, model, qinfo),
            expected_output
        );
    }

    println!("Multiple unpadded compatibility test successful");
}

 */
