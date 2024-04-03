// In this trait, we closely follow the notation of the LegoSNARK paper
trait CPSNARK {
    const ARITY: usize;

    type CommitmentKeyType; // ck used to commit and open commitments
    type EvaluationKeyType; // ek used to produce proofs
    type VerificationKeyType; // vk used rto verify proofs

    type CommitmentType; // C
    type OpeningType; // O (a.k.a. opening hint)

    type InstanceType; // D_x
    type ValueType; // D containing D_i for 1 <= i <= arity
    type WitnessType; // D_omega

    type ProofType; // pi

    // All of the Vec below should have fixed size, e.g. an associated constant
    //     const arity: usize;
    // Unfortunately, the ability to write expressions such as
    //      commitments: [Self::CommitmentType; Self::arity]
    // is not supported in stable Rust yet (some nightly features are available)
    // which could help

    fn key_gen(
        ck: &Self::CommitmentKeyType,
        r: dyn Fn(&Self::InstanceType, &[Self::ValueType; Self::ARITY], &Self::WitnessType) -> bool,
    ) -> (Self::EvaluationKeyType, Self::VerificationKeyType);
    fn prove(
        ek: &Self::EvaluationKeyType,
        x: &Self::InstanceType,
        commitments: &[Self::CommitmentType; Self::ARITY],
        values: &[Self::ValueType; Self::ARITY],
        omega: &Self::WitnessType,
    ) -> Self::ProofType;
    fn verify_proof(
        vk: &Self::VerificationKeyType,
        x: &Self::InstanceType,
        commitments: &[Self::CommitmentType; Self::ARITY],
        pi: &Self::ProofType,
    ) -> bool;
}
