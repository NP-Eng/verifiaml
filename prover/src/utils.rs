use hcs_common::{
    BMMNode, Model, NodeOpsNative, Poly, ReLUNode, RequantizeBMMFloatNode, ReshapeNode, SmallNIO,
};

use crate::{NodeOpsProve, ProvableModel};
use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly_commit::PolynomialCommitment;

pub fn model_downcast<
    F: PrimeField + Absorb + From<ST> + From<ST::LT>,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
    ST: SmallNIO,
>(
    model: &Model<ST>,
) -> ProvableModel<F, S, PCS, ST> {
    let prove_nodes: Vec<Box<dyn NodeOpsProve<F, S, PCS, ST>>> = model
        .nodes
        .iter()
        .map(|x| node_downcast(x.as_ref()))
        .collect();

    ProvableModel::<F, S, PCS, ST> {
        nodes: prove_nodes,
        input_shape: model.input_shape.clone(),
        output_shape: model.output_shape.clone(),
    }
}

pub fn node_downcast<
    F: PrimeField + Absorb + From<ST> + From<ST::LT>,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
    ST: SmallNIO,
>(
    node: &dyn NodeOpsNative<ST>,
) -> Box<dyn NodeOpsProve<F, S, PCS, ST>> {
    let node_as_any = node.as_any();

    Option::None
        .or_else(|| {
            node_as_any
                .downcast_ref::<ReLUNode<ST>>()
                .map(|x| Box::new(x.clone()) as Box<dyn NodeOpsProve<F, S, PCS, ST>>)
        })
        .or_else(|| {
            node_as_any
                .downcast_ref::<ReshapeNode>()
                .map(|x| Box::new(x.clone()) as Box<dyn NodeOpsProve<F, S, PCS, ST>>)
        })
        .or_else(|| {
            node_as_any
                .downcast_ref::<BMMNode<ST>>()
                .map(|x| Box::new(x.clone()) as Box<dyn NodeOpsProve<F, S, PCS, ST>>)
        })
        .or_else(|| {
            node_as_any
                .downcast_ref::<RequantizeBMMFloatNode<ST>>()
                .map(|x| Box::new(x.clone()) as Box<dyn NodeOpsProve<F, S, PCS, ST>>)
        })
        .expect("No implementor of NodeOpsProve was received")
}

#[cfg(all(test, feature = "test-types"))]
mod test {
    use super::*;

    use {
        ark_bn254::Fr, ark_crypto_primitives::sponge::poseidon::PoseidonSponge, hcs_common::Ligero,
        hcs_common::Tensor,
    };

    fn get_nodes() -> Vec<Box<dyn NodeOpsNative<i8>>> {
        vec![
            Box::new(ReLUNode::<i8>::new(1, 1)),
            Box::new(ReshapeNode::new(vec![1], vec![1])),
            Box::new(BMMNode::<i8>::new(
                Tensor::new(vec![1], vec![1, 1]),
                Tensor::new(vec![1], vec![1]),
                1,
            )),
            Box::new(RequantizeBMMFloatNode::<i8>::new(1, 1.0, 1, 1.0, 1, 1.0, 1)),
        ]
    }

    #[test]
    fn test_downcast_nodes() {
        let nodes = get_nodes();

        let nodes_proof: Vec<Box<dyn NodeOpsProve<Fr, PoseidonSponge<Fr>, Ligero<Fr>, i8>>> = nodes
            .iter()
            .map(|x| node_downcast::<Fr, PoseidonSponge<Fr>, Ligero<Fr>, i8>(x.as_ref()))
            .collect();

        assert!(nodes_proof
            .iter()
            .zip(nodes.iter())
            .all(|(x, y)| { x.type_name() == y.type_name() }));
    }

    #[test]
    fn test_downcast_into_provable_model() {
        let model = Model::<i8> {
            nodes: get_nodes(),
            input_shape: vec![1],
            output_shape: vec![1],
        };

        let _: ProvableModel<Fr, PoseidonSponge<Fr>, Ligero<Fr>, i8> = model_downcast(&model);
    }
}
