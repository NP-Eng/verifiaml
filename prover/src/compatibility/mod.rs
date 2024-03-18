// TODO: Compare all 10000 MNSIT images run with the manual model and the
// rust model -> Expected 100% accuracy

mod tests {
    use hcs_common::QArray;
    use pyo3::prelude::*;
    
    const PERCEPTRON_PATH: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../exploring_tf_lite/q_model_runner.py"
    ));

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

    #[test]
    fn test_get_two_layer_perceptron_mnist_output() {
        let expected_output =
            QArray::read("examples/two_layer_perceptron_mnist/data/output_test_150.json");
        assert_eq!(
            get_model_output("two_layer_perceptron_mnist", 150),
            expected_output
        );
    }

    #[test]
    fn test_get_simple_perceptron_mnist_output() {
        let expected_output =
            QArray::read("examples/simple_perceptron_mnist/data/output_test_150.json");
        assert_eq!(
            get_model_output("simple_perceptron_mnist", 150),
            expected_output
        );
    }
}

// TODO: Compare the output of the manual model and the tflite model -> print
// and compare
