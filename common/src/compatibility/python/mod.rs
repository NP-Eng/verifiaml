#[cfg(test)]
mod tests;

use pyo3::{prelude::*, PyAny};

use crate::QArray;

const PERCEPTRON_PATH: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../exploring_tf_lite/q_model_runner.py"
));

pub fn get_model(py: Python, model_name: &str, args: Option<Vec<(&str, &str)>>) -> Py<PyAny> {
    let func: Py<PyAny> = PyModule::from_code_bound(py, PERCEPTRON_PATH, "", "")
        .unwrap()
        .getattr("get_model")
        .unwrap()
        .into();
    func.call1(py, (model_name, args)).unwrap()
}

pub fn save_model_parameters_as_qarray(py: Python, model: &Py<PyAny>, path: &str) {
    model
        .call_method1(py, "save_params_as_qarray", (path,))
        .unwrap();
}

pub fn get_model_input<'py, T>(
    python: Python<'py>,
    model: &Py<PyAny>,
    index: Option<usize>,
) -> QArray<f32>
where
    T: Into<QArray<f32>> + FromPyObject<'py> + Clone,
{
    let result = model.call_method1(python, "get_input", (index.unwrap_or(150),));

    // Downcast the result to the expected type
    let model_input = result.unwrap().extract::<T>(python).unwrap();

    model_input.into()
}

pub fn get_model_output(py: Python, model: &Py<PyAny>, index: Option<usize>) -> QArray<u8> {
    let result = model.call_method1(py, "get_output", (index.unwrap_or(150),));

    // Downcast the result to the expected type
    let model_output = result.unwrap().extract::<Vec<u8>>(py).unwrap();

    QArray::from(model_output)
}
