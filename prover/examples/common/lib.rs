use hcs_common::{quantise_f32_u8_nne, InferenceProof, Model, Poly, Tensor};
use hcs_prover::ProveModel;

use hcs_verifier::VerifyModel;

use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly_commit::PolynomialCommitment;
use ark_std::test_rng;

pub fn prove_inference<F, S, PCS>(
    input_path: &str,
    expected_output_path: &str,
    model: &Model<i8>,
    qinfo: (f32, u8),
    sponge: S,
    output_shape: Vec<usize>,
) where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    let input: Tensor<f32> = Tensor::read(input_path);
    let expected_output: Tensor<u8> = Tensor::read(expected_output_path);

    let quantised_input: Tensor<u8> = Tensor::new(
        quantise_f32_u8_nne(input.values(), qinfo.0, qinfo.1),
        input.shape().clone(),
    );

    let input_i8 = (quantised_input.cast::<i32>() - 128).cast::<i8>();

    let mut sponge = sponge;

    let mut rng = test_rng();
    let (ck, _) = model.setup_keys::<F, S, PCS, _>(&mut rng).unwrap();

    let (node_coms, node_com_states): (Vec<_>, Vec<_>) =
        model.commit(&ck, None).into_iter().unzip();

    let inference_proof: InferenceProof<F, S, PCS, i8> = model.prove_inference(
        &ck,
        Some(&mut rng),
        &mut sponge,
        &node_coms,
        &node_com_states,
        input_i8,
    );

    let output_qtypearray = inference_proof.outputs[0].clone();

    let output_i8 = output_qtypearray.unwrap_small();

    let output_u8: Tensor<u8> = (output_i8.cast::<i32>() + 128).cast();

    assert_eq!(output_u8.compact_resize(output_shape, 0), expected_output);

    println!("Inference proof test successful");
}

pub fn verify_inference<F, S, PCS>(
    input_path: &str,
    expected_output_path: &str,
    model: &Model<i8>,
    qinfo: (f32, u8),
    sponge: S,
    output_shape: Vec<usize>,
) where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    let input: Tensor<f32> = Tensor::read(input_path);
    let expected_output: Tensor<u8> = Tensor::read(expected_output_path);

    let quantised_input: Tensor<u8> = Tensor::new(
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

    let inference_proof: InferenceProof<F, S, PCS, i8> = model.prove_inference(
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
