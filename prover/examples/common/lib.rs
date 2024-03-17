use hcs_common::{quantise_f32_u8_nne, InferenceProof, Model, Poly, QArray};
use hcs_prover::ProveModel;

use hcs_verifier::VerifyModel;

use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly_commit::PolynomialCommitment;
use ark_std::test_rng;

// Auxiliary function
fn unpadded_inference<F, S, PCS>(
    raw_input: QArray<f32>,
    model: &Model<i8, i32>,
    qinfo: (f32, u8),
) -> QArray<u8>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    let quantised_input: QArray<u8> = QArray::new(
        quantise_f32_u8_nne(raw_input.values(), qinfo.0, qinfo.1),
        raw_input.shape().clone(),
    );

    let input_i8 = (quantised_input.cast::<i32>() - 128).cast::<i8>();

    let output_i8 = model.evaluate(input_i8);

    (output_i8.cast::<i32>() + 128).cast()
}

// Auxiliary function
fn padded_inference<F, S, PCS>(
    raw_input: QArray<f32>,
    model: &Model<i8, i32>,
    qinfo: (f32, u8),
) -> QArray<u8>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    let quantised_input: QArray<u8> = QArray::new(
        quantise_f32_u8_nne(raw_input.values(), qinfo.0, qinfo.1),
        raw_input.shape().clone(),
    );

    let input_i8 = (quantised_input.cast::<i32>() - 128).cast::<i8>();

    let output_i8 =
        <Model<i8, i32> as ProveModel<F, S, PCS, i8, i32>>::padded_evaluate(model, input_i8);

    (output_i8.cast::<i32>() + 128).cast()
}

pub fn run_unpadded<F, S, PCS>(
    input_path: &str,
    expected_output_path: &str,
    model: &Model<i8, i32>,
    qinfo: (f32, u8),
) where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    let raw_input: QArray<f32> = QArray::read(input_path);
    let expected_output: QArray<u8> = QArray::read(expected_output_path);

    let output_u8 = unpadded_inference::<F, S, PCS>(raw_input, model, qinfo);

    assert_eq!(output_u8, expected_output);

    println!("Single unpadded compatibility test successful");
}

pub fn run_padded<F, S, PCS>(
    input_path: &str,
    expected_output_path: &str,
    model: &Model<i8, i32>,
    qinfo: (f32, u8),
) where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    let raw_input: QArray<f32> = QArray::read(input_path);
    let expected_output: QArray<u8> = QArray::read(expected_output_path);

    let output_u8 = padded_inference::<F, S, PCS>(raw_input, model, qinfo);

    assert_eq!(output_u8, expected_output);

    println!("Single padded compatibility test successful");
}

pub fn multi_run_unpadded<F, S, PCS>(
    inputs_path: &str,
    expected_outputs_path: &str,
    model: &Model<i8, i32>,
    qinfo: (f32, u8),
) where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    let raw_inputs: Vec<QArray<f32>> = QArray::read_list(inputs_path);
    let expected_outputs: Vec<QArray<u8>> = QArray::read_list(expected_outputs_path);

    for (raw_input, expected_output) in raw_inputs.into_iter().zip(expected_outputs.into_iter()) {
        assert_eq!(
            unpadded_inference::<F, S, PCS>(raw_input, model, qinfo),
            expected_output
        );
    }

    println!("Multiple unpadded compatibility test successful");
}

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
    let raw_inputs: Vec<QArray<f32>> = QArray::read_list(inputs_path);
    let expected_outputs: Vec<QArray<u8>> = QArray::read_list(expected_outputs_path);

    for (raw_input, expected_output) in raw_inputs.into_iter().zip(expected_outputs.into_iter()) {
        assert_eq!(
            padded_inference::<F, S, PCS>(raw_input, model, qinfo),
            expected_output
        );
    }

    println!("Multiple unpadded compatibility test successful");
}

pub fn prove_inference<F, S, PCS>(
    input_path: &str,
    expected_output_path: &str,
    model: &Model<i8, i32>,
    qinfo: (f32, u8),
    sponge: S,
    output_shape: Vec<usize>,
) where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    let input: QArray<f32> = QArray::read(input_path);
    let expected_output: QArray<u8> = QArray::read(expected_output_path);

    let quantised_input: QArray<u8> = QArray::new(
        quantise_f32_u8_nne(input.values(), qinfo.0, qinfo.1),
        input.shape().clone(),
    );

    let input_i8 = (quantised_input.cast::<i32>() - 128).cast::<i8>();

    let mut sponge = sponge;

    let mut rng = test_rng();
    let (ck, _) = model.setup_keys::<F, S, PCS, _>(&mut rng).unwrap();

    let (node_coms, node_com_states): (Vec<_>, Vec<_>) =
        model.commit(&ck, None).into_iter().unzip();

    let inference_proof: InferenceProof<F, S, PCS, i8, i32> = model.prove_inference(
        &ck,
        Some(&mut rng),
        &mut sponge,
        &node_coms,
        &node_com_states,
        input_i8,
    );

    let output_qtypearray = inference_proof.outputs[0].clone();

    let output_i8 = output_qtypearray.unwrap_small();

    let output_u8: QArray<u8> = (output_i8.cast::<i32>() + 128).cast();

    assert_eq!(output_u8.compact_resize(output_shape, 0), expected_output);

    println!("Inference proof test successful");
}

pub fn verify_inference<F, S, PCS>(
    input_path: &str,
    expected_output_path: &str,
    model: &Model<i8, i32>,
    qinfo: (f32, u8),
    sponge: S,
    output_shape: Vec<usize>,
) where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    let input: QArray<f32> = QArray::read(input_path);
    let expected_output: QArray<u8> = QArray::read(expected_output_path);

    let quantised_input: QArray<u8> = QArray::new(
        quantise_f32_u8_nne(input.values(), qinfo.0, qinfo.1),
        input.shape().clone(),
    );

    let input_i8 = (quantised_input.cast::<i32>() - 128).cast::<i8>();

    // Cloning the initial state of the sponge to start proof and verification
    // with the same fresh sponge
    let mut proving_sponge = sponge.clone();
    let mut verification_sponge = sponge;

    let mut rng = test_rng();
    let (ck, vk) = model.setup_keys::<F, S, PCS, _>(&mut rng).unwrap();

    let (node_coms, node_com_states): (Vec<_>, Vec<_>) =
        model.commit(&ck, None).into_iter().unzip();

    let inference_proof: InferenceProof<F, S, PCS, i8, i32> = model.prove_inference(
        &ck,
        Some(&mut rng),
        &mut proving_sponge,
        &node_coms,
        &node_com_states,
        input_i8,
    );

    let output_qtypearray = inference_proof.outputs[0].clone();

    assert!(model.verify_inference(&vk, &mut verification_sponge, &node_coms, inference_proof));

    let output_i8 = output_qtypearray.unwrap_small();

    let output_u8 = (output_i8.cast::<i32>() + 128).cast::<u8>();

    assert_eq!(output_u8.compact_resize(output_shape, 0), expected_output);

    println!("Inference verification test successful");
}
