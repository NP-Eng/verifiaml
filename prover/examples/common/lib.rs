use hcs_common::{quantise_f32_u8_nne, Model, Poly, QArray, QSmallType, QTypeArray};
use hcs_prover::ProveModel;

use hcs_verifier::VerifyModel;

use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly_commit::PolynomialCommitment;
use ark_std::test_rng;

pub fn prove_inference<F, S, PCS>(
    input_path: &str,
    expected_output_path: &str,
    model: &Model<F, S, PCS>,
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

    let input_i8 = (quantised_input.cast::<i32>() - 128).cast::<QSmallType>();

    let mut sponge = sponge;

    let mut rng = test_rng();
    let (ck, _) = model.setup_keys(&mut rng).unwrap();

    let (node_coms, node_com_states): (Vec<_>, Vec<_>) =
        model.commit(&ck, None).into_iter().unzip();

    let inference_proof = model.prove_inference(
        &ck,
        Some(&mut rng),
        &mut sponge,
        &node_coms,
        &node_com_states,
        input_i8,
    );

    let output_qtypearray = inference_proof.inputs[0].clone();

    let output_i8 = match output_qtypearray {
        QTypeArray::S(o) => o,
        _ => panic!("Expected QTypeArray::S"),
    };

    let output_u8: QArray<u8> = (output_i8.cast::<i32>() + 128).cast();

    assert_eq!(output_u8.compact_resize(output_shape, 0), expected_output);

    println!("Inference proof test successful");
}

pub fn verify_inference<F, S, PCS>(
    input_path: &str,
    expected_output_path: &str,
    model: &Model<F, S, PCS>,
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

    let input_i8 = (quantised_input.cast::<i32>() - 128).cast::<QSmallType>();

    // Cloning the initial state of the sponge to start proof and verification
    // with the same fresh sponge
    let mut proving_sponge = sponge.clone();
    let mut verification_sponge = sponge;

    let mut rng = test_rng();
    let (ck, vk) = model.setup_keys(&mut rng).unwrap();

    let (node_coms, node_com_states): (Vec<_>, Vec<_>) =
        model.commit(&ck, None).into_iter().unzip();

    let inference_proof = model.prove_inference(
        &ck,
        Some(&mut rng),
        &mut proving_sponge,
        &node_coms,
        &node_com_states,
        input_i8,
    );

    let output_qtypearray = inference_proof.outputs[0].clone();

    assert!(model.verify_inference(&vk, &mut verification_sponge, &node_coms, inference_proof));

    let output_i8 = match output_qtypearray {
        QTypeArray::S(o) => o,
        _ => panic!("Expected QTypeArray::S"),
    };

    let output_u8 = (output_i8.cast::<i32>() + 128).cast::<u8>();

    assert_eq!(output_u8.compact_resize(output_shape, 0), expected_output);

    println!("Inference verification test successful");
}
