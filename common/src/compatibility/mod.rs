pub mod example_models;

#[cfg(all(test, feature = "python"))]
mod tests {
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
        quantise_f32_u8_nne, Ligero, Model, QArray,
    };
    use ark_bn254::Fr;
    use ark_crypto_primitives::sponge::poseidon::PoseidonSponge;
    use more_asserts::*;

    use pyo3::{prelude::*, PyAny};

    const PERCEPTRON_PATH: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../exploring_tf_lite/q_model_runner.py"
    ));

    const NB_OUTPUTS: usize = 1000;

    // TODO: We allow incorrect outputs because the quantisation from tf lite
    // is inexact. We should fix this in the future. Currently, the outputs are
    // within the allowed error.
    const ALLOWED_ERROR_MARGIN: f32 = 0.1;

    fn get_model(py: Python, model_name: &str) -> Py<PyAny> {
        let func: Py<PyAny> = PyModule::from_code(py, PERCEPTRON_PATH, "", "")
            .unwrap()
            .getattr("get_model")
            .unwrap()
            .into();
        func.call1(py, (model_name,)).unwrap()
    }

    fn get_model_input(py: Python, model: &Py<PyAny>, index: Option<usize>) -> QArray<f32> {
        let result = model.call_method1(py, "get_input", (index.unwrap_or(150),));

        // Downcast the result to the expected type
        let model_input = result.unwrap().extract::<Vec<Vec<f32>>>(py).unwrap();

        QArray::from(model_input)
    }

    fn get_model_output(py: Python, model: &Py<PyAny>, index: Option<usize>) -> QArray<u8> {
        let result = model.call_method1(py, "get_output", (index.unwrap_or(150),));

        // Downcast the result to the expected type
        let model_output = result.unwrap().extract::<Vec<u8>>(py).unwrap();

        QArray::from(model_output)
    }

    fn unpadded_inference(
        raw_input: QArray<f32>,
        model: &Model<i8, i32>,
        qinfo: (f32, u8),
    ) -> QArray<u8> {
        let quantised_input: QArray<u8> = QArray::new(
            quantise_f32_u8_nne(raw_input.values(), qinfo.0, qinfo.1),
            raw_input.shape().clone(),
        );

        let input_i8 = (quantised_input.cast::<i32>() - 128).cast::<i8>();

        let output_i8 = model.evaluate(input_i8);

        (output_i8.cast::<i32>() + 128).cast()
    }

    #[test]
    fn test_simple_perceptron_mnist_single_input() {
        let expected_input =
            QArray::read("examples/simple_perceptron_mnist/data/input_test_150.json");
        assert_eq!(
            Python::with_gil(|py| get_model_input(py, &get_model(py, "QSimplePerceptron"), None)),
            expected_input
        );
    }

    #[test]
    fn test_two_layer_perceptron_mnist_single_input() {
        let expected_input =
            QArray::read("examples/two_layer_perceptron_mnist/data/input_test_150.json");
        assert_eq!(
            Python::with_gil(|py| get_model_input(py, &get_model(py, "QTwoLayerPerceptron"), None)),
            expected_input
        );
    }

    #[test]
    fn test_simple_perceptron_mnist_single_output() {
        let expected_output =
            QArray::read("examples/simple_perceptron_mnist/data/output_test_150.json");
        assert_eq!(
            Python::with_gil(|py| get_model_output(py, &get_model(py, "QSimplePerceptron"), None)),
            expected_output
        );
    }

    #[test]
    fn test_two_layer_perceptron_mnist_single_output() {
        let expected_output =
            QArray::read("examples/two_layer_perceptron_mnist/data/output_test_150.json");
        assert_eq!(
            Python::with_gil(|py| get_model_output(
                py,
                &get_model(py, "QTwoLayerPerceptron"),
                None
            )),
            expected_output
        );
    }

    #[test]
    fn test_two_layer_perceptron_mnist_all_outputs() {
        let two_layer_perceptron_mnist =
            build_two_layer_perceptron_mnist::<Fr, PoseidonSponge<Fr>, Ligero<Fr>>(false);

        let correct_samples: usize = Python::with_gil(|py| {
            let tf_lite_model = get_model(py, "QTwoLayerPerceptron");
            (0..NB_OUTPUTS)
                .into_iter()
                .map(|i| {
                    let raw_input = get_model_input(py, &tf_lite_model, Some(i));
                    let expected_output = get_model_output(py, &tf_lite_model, Some(i));

                    let output = unpadded_inference(
                        raw_input,
                        &two_layer_perceptron_mnist,
                        (
                            S_INPUT_TWO_LAYER_PERCEPTRON_MNIST,
                            Z_INPUT_TWO_LAYER_PERCEPTRON_MNIST,
                        ),
                    );

                    (output == expected_output) as usize
                })
                .sum()
        });

        println!(
            "Two-layer perceptron discrepancies: {} out of {}",
            NB_OUTPUTS - correct_samples,
            NB_OUTPUTS
        );

        assert_ge!(
            correct_samples as f32 / NB_OUTPUTS as f32,
            1.0 - ALLOWED_ERROR_MARGIN
        );
    }

    #[test]
    fn test_simple_perceptron_mnist_all_outputs() {
        let simple_perceptron_mnist =
            build_simple_perceptron_mnist::<Fr, PoseidonSponge<Fr>, Ligero<Fr>>(false);

        let correct_samples: usize = Python::with_gil(|py| {
            let tf_lite_model = get_model(py, "QSimplePerceptron");
            (0..NB_OUTPUTS)
                .into_iter()
                .map(|i| {
                    let raw_input = get_model_input(py, &tf_lite_model, Some(i));
                    let expected_output = get_model_output(py, &tf_lite_model, Some(i));

                    let output = unpadded_inference(
                        raw_input,
                        &simple_perceptron_mnist,
                        (
                            S_INPUT_SIMPLE_PERCEPTRON_MNIST,
                            Z_INPUT_SIMPLE_PERCEPTRON_MNIST,
                        ),
                    );

                    (output == expected_output) as usize
                })
                .sum()
        });

        println!(
            "Simple perceptron discrepancies: {} out of {}",
            NB_OUTPUTS - correct_samples,
            NB_OUTPUTS
        );

        assert_ge!(
            correct_samples as f32 / NB_OUTPUTS as f32,
            1.0 - ALLOWED_ERROR_MARGIN
        );
    }
}
