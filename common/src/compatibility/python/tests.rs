use super::*;

use crate::{
    compatibility::example_models::{
        simple_perceptron_mnist::{
            build_simple_perceptron_mnist,
            parameters::{
                S_INPUT as S_INPUT_SIMPLE_PERCEPTRON_MNIST,
                Z_INPUT as Z_INPUT_SIMPLE_PERCEPTRON_MNIST,
            },
        },
        two_layer_perceptron_mnist::{
            build_two_layer_perceptron_mnist,
            parameters::{
                S_INPUT as S_INPUT_TWO_LAYER_PERCEPTRON_MNIST,
                Z_INPUT as Z_INPUT_TWO_LAYER_PERCEPTRON_MNIST,
            },
        },
    },
    quantise_f32_u8_nne, BMMRequantizationStrategy, Ligero, Model, Tensor,
};
use ark_bn254::Fr;
use ark_crypto_primitives::sponge::poseidon::PoseidonSponge;
use more_asserts::*;

const NB_OUTPUTS: usize = 10000;

// TODO: We allow incorrect outputs because the quantisation from tf lite
// is inexact. We should fix this in the future. Currently, the outputs are
// within the allowed error.
const ALLOWED_ERROR_MARGIN: f32 = 0.1;

fn unpadded_inference(raw_input: Tensor<f32>, model: &Model<i8>, qinfo: (f32, u8)) -> Tensor<u8> {
    let quantised_input: Tensor<u8> = Tensor::new(
        quantise_f32_u8_nne(raw_input.values(), qinfo.0, qinfo.1),
        raw_input.shape().clone(),
    );

    let input_i8 = (quantised_input.cast::<i32>() - 128).cast::<i8>();

    let output_i8 = model.evaluate(input_i8);

    (output_i8.cast::<i32>() + 128).cast()
}

fn run_model_all_outputs(model_name: &str, req_strategy: BMMRequantizationStrategy) {
    let correct_samples: usize = Python::with_gil(|py| {
        let (s_input, z_input, rust_model) = match model_name {
            "QSimplePerceptron" => (
                S_INPUT_SIMPLE_PERCEPTRON_MNIST,
                Z_INPUT_SIMPLE_PERCEPTRON_MNIST,
                build_simple_perceptron_mnist::<Fr, PoseidonSponge<Fr>, Ligero<Fr>>(req_strategy),
            ),
            "QTwoLayerPerceptron" => (
                S_INPUT_TWO_LAYER_PERCEPTRON_MNIST,
                Z_INPUT_TWO_LAYER_PERCEPTRON_MNIST,
                build_two_layer_perceptron_mnist::<Fr, PoseidonSponge<Fr>, Ligero<Fr>>(
                    req_strategy,
                ),
            ),
            _ => panic!("Model not found"),
        };
        let tf_lite_model = get_model(py, model_name, None);
        (0..NB_OUTPUTS)
            .into_iter()
            .map(|i| {
                let raw_input = get_model_input::<Vec<Vec<f32>>>(py, &tf_lite_model, i);
                let expected_output = get_model_output(py, &tf_lite_model, i);
                let output = unpadded_inference(raw_input, &rust_model, (s_input, z_input));
                (output == expected_output) as usize
            })
            .sum()
    });

    println!(
        "{} with requantization strategy {:?}, discrepancies: {} out of {}",
        model_name,
        req_strategy,
        NB_OUTPUTS - correct_samples,
        NB_OUTPUTS
    );

    assert_ge!(
        correct_samples as f32 / NB_OUTPUTS as f32,
        1.0 - ALLOWED_ERROR_MARGIN
    );
}

#[test]
fn test_get_model_input() {
    let expected_input = Tensor::read("examples/simple_perceptron_mnist/data/input_test_150.json");
    assert_eq!(
        Python::with_gil(|py| get_model_input::<Vec<Vec<f32>>>(
            py,
            &get_model(py, "QSimplePerceptron", None),
            150
        )),
        expected_input
    );
}

#[test]
fn test_simple_perceptron_mnist_single_output() {
    let expected_output =
        Tensor::read("examples/simple_perceptron_mnist/data/output_test_150.json");
    assert_eq!(
        Python::with_gil(|py| get_model_output(py, &get_model(py, "QSimplePerceptron", None), 150)),
        expected_output
    );
}

#[test]
fn test_two_layer_perceptron_mnist_single_output() {
    let expected_output =
        Tensor::read("examples/two_layer_perceptron_mnist/data/output_test_150.json");
    assert_eq!(
        Python::with_gil(|py| get_model_output(
            py,
            &get_model(py, "QTwoLayerPerceptron", None),
            150
        )),
        expected_output
    );
}

#[test]
fn test_simple_perceptron_req_float() {
    run_model_all_outputs("QSimplePerceptron", BMMRequantizationStrategy::Floating);
}

#[test]
fn test_two_layer_perceptron_req_float() {
    run_model_all_outputs("QTwoLayerPerceptron", BMMRequantizationStrategy::Floating);
}

#[test]
fn test_simple_perceptron_req_ref() {
    run_model_all_outputs("QSimplePerceptron", BMMRequantizationStrategy::Reference);
}

#[test]
fn test_two_layer_perceptron_req_ref() {
    run_model_all_outputs("QTwoLayerPerceptron", BMMRequantizationStrategy::Reference);
}

#[test]
fn test_simple_perceptron_req_single() {
    run_model_all_outputs("QSimplePerceptron", BMMRequantizationStrategy::SingleRound);
}

#[test]
fn test_two_layer_perceptron_req_single() {
    run_model_all_outputs(
        "QTwoLayerPerceptron",
        BMMRequantizationStrategy::SingleRound,
    );
}
