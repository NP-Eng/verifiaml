use crate::{Node, NodeCommitment};

// In this trait, we closely follow the notation of the LegoSNARK paper
trait CPSNARK {
    const ARITY: usize;

    type CommitmentKey; // ck used to commit and open commitments
    type EvaluationKey; // ek used to produce proofs
    type VerificationKey; // vk used to verify proofs

    type Commitment; // C
    type Opening; // O (a.k.a. opening hint)

    type Instance; // D_x
    type Value; // D containing D_i for 1 <= i <= arity
    type Witness; // D_omega

    type Proof; // pi

    // A description of a relation on Instance x (Value x Witness)
    // which is used to generate the evaluation and verification keys
    type Relation;

    fn key_gen(
        ck: &Self::CommitmentKey,
        r: &Self::Relation,
    ) -> (Self::EvaluationKey, Self::VerificationKey);

    fn prove(
        ek: &Self::EvaluationKey,
        x: &Self::Instance,
        commitments: &[Self::Commitment; Self::ARITY],
        values: &[Self::Value; Self::ARITY],
        openings: &[Self::Opening; Self::ARITY],
        omega: &Self::Witness,
    ) -> Self::Proof;

    fn verify_proof(
        vk: &Self::VerificationKey,
        x: &Self::Instance,
        commitments: &[Self::Commitment; Self::ARITY],
        pi: &Self::Proof,
    ) -> bool;
}

trait NodeCPSNARK {
    type CommitmentKey; // ck used to commit and open commitments to parameters, inputs and outputs
    type EvaluationKey; // ek used to produce proofs
    type VerificationKey; // vk used to verify proofs

    type ParamCommitment; // C_1
    type ParamHint; // O_1 (a.k.a. opening hint)
    type ParamValue; // D_1

    type IOCommitment; // C_2, space for the input as well as output
    type IOHint; // O_2 (a.k.a. opening hint)
    type IOValue; // D_2

    type Instance; // D_x

    type Proof; // pi

    // A description of a relation on Instance x (Value x Witness)
    // which is used to generate the evaluation and verification keys
    type Relation;

    fn get_param_commmitment(&self) -> &Self::ParamCommitment;

    fn get_param_value(&self) -> &Self::ParamValue;

    fn get_param_hint(&self) -> &Self::ParamHint;

    fn key_gen(&self, ck: &Self::CommitmentKey) -> (Self::EvaluationKey, Self::VerificationKey);

    fn prove(
        &self,
        ek: &Self::EvaluationKey,
        // the instance is read from self
        input_commitment: &Self::IOCommitment,
        ouput_commitment: &Self::IOCommitment,
        // the param commitment is read from self
        input_value: &Self::IOValue,
        output_value: &Self::IOValue,
        // the param value(s) are read from self
        input_hint: &Self::IOHint,
        output_hint: &Self::IOHint,
        // the param hint is read from self
        // no non-committed witness omega
    ) -> Self::Proof;

    fn verify_proof(
        &self,
        vk: &Self::VerificationKey,
        // the instance is read from self
        input_commitment: &Self::IOCommitment,
        ouput_commitment: &Self::IOCommitment,
        // the param commitment is read from self
        pi: &Self::Proof,
    ) -> bool;
}

enum ParamOrIO<ParamType, IOType> {
    Param(ParamType),
    IO(IOType),
}

impl<NCPS: NodeCPSNARK> CPSNARK for NCPS {
    const ARITY: usize = 3;

    type CommitmentKey = NCPS::CommitmentKey;
    type EvaluationKey = NCPS::EvaluationKey;
    type VerificationKey = NCPS::VerificationKey;

    type Commitment = ParamOrIO<NCPS::ParamCommitment, NCPS::IOCommitment>;

    type Opening = ParamOrIO<NCPS::ParamHint, NCPS::IOHint>;

    type Value = ParamOrIO<NCPS::ParamValue, NCPS::IOValue>;

    type Instance = NCPS::Instance;

    type Witness = ();

    type Proof = NCPS::Proof;

    type Relation = NCPS::Relation;

    fn key_gen(
        ck: &Self::CommitmentKey,
        r: &Self::Relation,
    ) -> (Self::EvaluationKey, Self::VerificationKey) {
        todo!()
    }

    fn prove(
        ek: &Self::EvaluationKey,
        x: &Self::Instance,
        commitments: &[Self::Commitment; Self::ARITY],
        values: &[Self::Value; Self::ARITY],
        openings: &[Self::Opening; Self::ARITY],
        omega: &Self::Witness,
    ) -> Self::Proof {
        todo!()
    }

    fn verify_proof(
        vk: &Self::VerificationKey,
        x: &Self::Instance,
        commitments: &[Self::Commitment; Self::ARITY],
        pi: &Self::Proof,
    ) -> bool {
        todo!()
    }
}
