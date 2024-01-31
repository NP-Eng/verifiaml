
use ark_ff::PrimeField;
use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_poly_commit::PolynomialCommitment;

use crate::model::{
    Model,
    Poly,
    nodes::{
        Node,
        reshape::ReshapeNode,
        fc::FCNode,
        relu::ReLUNode,
    },
};

mod parameters;

use parameters::*;

const INPUT_DIMS: &[usize] = &[28, 28];
const OUTPUT_DIMS: &[usize] = &[10];

fn build_simple_perceptron_mnist<F, S, PCS>() -> Model<F, S, PCS> 
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{

    let flat_dim = INPUT_DIMS.iter().product();

    let reshape: ReshapeNode<F, S, PCS> = ReshapeNode::new(
        INPUT_DIMS.to_vec(),
        vec![flat_dim],
    );

    let fc: FCNode<F, S, PCS> = FCNode::new(
        WEIGHTS.to_vec(),
        BIAS.to_vec(),
        (flat_dim, OUTPUT_DIMS[0]),
        S_I,
        Z_I,
        S_W,
        Z_W,
        S_O,
        Z_O,
    );

    Model::new(vec![
        Node::Reshape(reshape),
        Node::FC(fc),
    ])
}

#[cfg(test)]
mod tests {
    use ark_crypto_primitives::sponge::poseidon::PoseidonSponge;    
    use ark_bn254::{Fr, G1Affine};
    use ark_poly_commit::hyrax::HyraxPC;

    use crate::model::Poly;
    
    type Sponge = PoseidonSponge<Fr>;
    type Hyrax254 = HyraxPC<G1Affine, Poly<Fr>, Sponge>;
    
    use super::*;

    #[test]
    fn run_simple_perceptron_mnist() {

        let input = vec![vec![30; 28]; 28];
        let expected_output: Vec<u8> = vec![];

        let perceptron = build_simple_perceptron_mnist::<Fr, Sponge, Hyrax254>();
        
        let input_i8 = input.into_iter().map(|r|
            r.into_iter().map(|x| ((x as i32) - 128) as i8).collect::<Vec<i8>>()
        ).collect::<Vec<Vec<i8>>>().into();

        let output = perceptron.evaluate(input_i8);

        let output_u8: Vec<u8> = output.values()[0][0].iter().map(|x| ((*x as i32) + 128) as u8).collect();

        // assert_eq!(output_u8, expected_output);
        println!("Output: {:?}", output_u8);
    }
}