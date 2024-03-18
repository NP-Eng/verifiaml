mod tests {
    use ark_bn254::Fr;
    use ark_crypto_primitives::sponge::poseidon::PoseidonSponge;
    use hcs_common::{
        example_models::{
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
                }
            }
        },
        quantise_f32_u8_nne, Ligero, Model, QArray, QSmallType,
    };

    use pyo3::prelude::*;

    const PERCEPTRON_PATH: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../exploring_tf_lite/q_model_runner.py"
    ));

    const NB_OUTPUTS: usize = 100;

    fn get_model_input(model_name: &str, index: usize) -> QArray<f32> {
        let model_input = Python::with_gil(|py| {
            let perceptron: Py<PyAny> = PyModule::from_code(py, PERCEPTRON_PATH, "", "")
                .unwrap()
                .getattr("get_model_input")
                .unwrap()
                .into();
            let result = perceptron.call1(py, (model_name, index));

            // Downcast the result to the expected type
            result.unwrap().extract::<Vec<Vec<f32>>>(py).unwrap()
        });
        QArray::from(model_input)
    }

    fn get_model_output(model_name: &str, index: usize) -> QArray<u8> {
        let model_output = Python::with_gil(|py| {
            let perceptron: Py<PyAny> = PyModule::from_code(py, PERCEPTRON_PATH, "", "")
                .unwrap()
                .getattr("get_model_output")
                .unwrap()
                .into();
            let result = perceptron.call1(py, (model_name, index));

            // Downcast the result to the expected type
            result.unwrap().extract::<Vec<u8>>(py).unwrap()
        });
        QArray::from(model_output)
    }

    fn unpadded_inference(
        raw_input: QArray<f32>,
        model: &Model<Fr, PoseidonSponge<Fr>, Ligero<Fr>>,
        qinfo: (f32, u8),
    ) -> QArray<u8> {
        let quantised_input: QArray<u8> = QArray::new(
            quantise_f32_u8_nne(raw_input.values(), qinfo.0, qinfo.1),
            raw_input.shape().clone(),
        );

        let input_i8 = (quantised_input.cast::<i32>() - 128).cast::<QSmallType>();

        let output_i8 = model.evaluate(input_i8);

        (output_i8.cast::<i32>() + 128).cast()
    }

    #[test]
    fn test_two_layer_perceptron_mnist_single_input() {
        let expected_input =
            QArray::read("examples/two_layer_perceptron_mnist/data/input_test_150.json");
        assert_eq!(
            get_model_input("two_layer_perceptron_mnist", 150),
            expected_input
        );
    }

    #[test]
    fn test_simple_perceptron_mnist_single_input() {
        let expected_input =
            QArray::read("examples/simple_perceptron_mnist/data/input_test_150.json");
        assert_eq!(
            get_model_input("simple_perceptron_mnist", 150),
            expected_input
        );
    }

    #[test]
    fn test_two_layer_perceptron_mnist_single_output() {
        let expected_output =
            QArray::read("examples/two_layer_perceptron_mnist/data/output_test_150.json");
        assert_eq!(
            get_model_output("two_layer_perceptron_mnist", 150),
            expected_output
        );
    }

    #[test]
    fn test_simple_perceptron_mnist_single_output() {
        let expected_output =
            QArray::read("examples/simple_perceptron_mnist/data/output_test_150.json");
        assert_eq!(
            get_model_output("simple_perceptron_mnist", 150),
            expected_output
        );
    }

    #[test]
    fn test_two_layer_perceptron_mnist_all_outputs() {
        let two_layer_perceptron_mnist: Model<Fr, PoseidonSponge<Fr>, Ligero<Fr>> =
            build_two_layer_perceptron_mnist();

        let correct_samples: usize = (0..NB_OUTPUTS)
            .into_iter()
            .map(|i| {
                let raw_input = get_model_input("two_layer_perceptron_mnist", i);
                let expected_output = get_model_output("two_layer_perceptron_mnist", i);

                let output =
                    unpadded_inference(raw_input, &two_layer_perceptron_mnist, (S_INPUT_TWO_LAYER_PERCEPTRON_MNIST, Z_INPUT_TWO_LAYER_PERCEPTRON_MNIST));

                (output == expected_output) as usize
            })
            .sum();

        assert_eq!(correct_samples, NB_OUTPUTS);
    }

    #[test]
    fn test_simple_perceptron_mnist_all_outputs() {
        let simple_perceptron_mnist: Model<Fr, PoseidonSponge<Fr>, Ligero<Fr>> =
            build_simple_perceptron_mnist();

        let correct_samples: usize = (0..NB_OUTPUTS)
            .into_iter()
            .map(|i| {
                let raw_input = get_model_input("simple_perceptron_mnist", i);
                let expected_output = get_model_output("simple_perceptron_mnist", i);

                let output =
                    unpadded_inference(raw_input, &simple_perceptron_mnist, (S_INPUT_SIMPLE_PERCEPTRON_MNIST, Z_INPUT_SIMPLE_PERCEPTRON_MNIST));

                (output == expected_output) as usize
            })
            .sum();

        assert_eq!(correct_samples, NB_OUTPUTS);
    }
}
