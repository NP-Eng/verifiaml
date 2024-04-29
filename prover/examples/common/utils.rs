use hcs_common::{BMMNode, Model, Poly, ReLUNode, RequantizeBMMFloatNode, ReshapeNode, SmallNIO};
use hcs_verifier::{NodeOpsVerify, VerifiableModel};

use hcs_prover::{NodeOpsProve, ProvableModel};
use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly_commit::PolynomialCommitment;

macro_rules! downcast_as {
    ($node:ident, $node_subtype:ty, $node_type:ty) => {
        $node
            .downcast_ref::<$node_type>()
            .map(|x| Box::new(x.clone()) as $node_subtype)
    };
}

macro_rules! downcast_to_either {
    ($node:ident, $node_subtype:ty, $node_type:ty, $($rest:ty),*) => {
        downcast_as!($node, $node_subtype, $node_type).or_else(|| downcast_to_either!($node, $node_subtype, $($rest),*))
    };
    ($node:ident, $node_subtype:ty, $node_type:ty) => {
        downcast_as!($node, $node_subtype, $node_type)
    };
}

macro_rules! node_downcast {
    ($node:expr, $small_type:ty, $node_subtype:ty) => {{
        let node_as_any = $node.as_any();
        downcast_to_either!(
            node_as_any,
            $node_subtype,
            // ADD NEW NODE TYPES HERE
            ReLUNode<$small_type>,
            ReshapeNode<$small_type>,
            BMMNode<$small_type>,
            RequantizeBMMFloatNode<$small_type>
        )
        .expect("Node type not supported")
    }};
}

macro_rules! model_downcast {
    ($model:expr, $model_subtype:ident, $node_subtype:ty, $small_type:ty) => {{
        let nodes_proof = $model
            .nodes
            .iter()
            .map(|node| node_downcast!(node, $small_type, $node_subtype))
            .collect();

        $model_subtype {
            nodes: nodes_proof,
            input_shape: $model.input_shape.clone(),
            output_shape: $model.output_shape.clone(),
        }
    }};
}

pub(crate) fn as_provable_model<
    F: PrimeField + Absorb + From<ST> + From<ST::LT>,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
    ST: SmallNIO,
>(
    model: &Model<ST>,
) -> ProvableModel<F, S, PCS, ST> {
    model_downcast!(
        model,
        ProvableModel,
        Box<dyn NodeOpsProve<F, S, PCS, ST>>,
        ST
    )
}

pub(crate) fn as_verifiable_model<
    F: PrimeField + Absorb + From<ST> + From<ST::LT>,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
    ST: SmallNIO,
>(
    model: &Model<ST>,
) -> VerifiableModel<F, S, PCS, i8> {
    model_downcast!(
        model,
        VerifiableModel,
        Box<dyn NodeOpsVerify<F, S, PCS, i8>>,
        i8
    )
}

#[cfg(all(test, feature = "test-types"))]
mod test {

    use super::*;

    use ark_bn254::Fr;
    use ark_crypto_primitives::sponge::poseidon::PoseidonSponge;
    use hcs_common::{Ligero, NodeOpsNative, Tensor};

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

        let nodes_proof: Vec<Box<_>> = nodes
            .iter()
            .map(|x| {
                node_downcast!(
                    x,
                    i8,
                    Box<dyn NodeOpsProve<Fr, PoseidonSponge<Fr>, Ligero<Fr>, i8>>
                )
            })
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

        let _ = as_provable_model::<Fr, PoseidonSponge<Fr>, Ligero<Fr>, i8>(&model);
    }

    #[test]
    fn test_downcast_into_verifiable_model() {
        let model = Model::<i8> {
            nodes: get_nodes(),
            input_shape: vec![1],
            output_shape: vec![1],
        };

        let _ = as_verifiable_model::<Fr, PoseidonSponge<Fr>, Ligero<Fr>, i8>(&model);
    }
}
