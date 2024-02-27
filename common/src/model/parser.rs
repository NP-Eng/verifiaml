
// Temporary file to parse simple models from strings. This will likely be
// replaced by an ONNX parser and/or serialisation capabilities. It is not
// intended to have robust error handling

use crate::model::Model;

fn parse_model(s: &str) -> Model {

    let layers = serde_json::from_str(s).iter().map(|l| {
        match l.layer_type {
            "relu" => parse_relu_layer(l),
            "matmul" => parse_matmul_layer(l),
            _ => panic!("Unknown layer type {}", l.layer_type),
        }
    }).collect();
}
