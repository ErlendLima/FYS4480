
        """
        Compute ⟨Φ_ij^ab|̄H|0⟩ = 0. Splitting up the operator, we have
        Ĥ =  Eref + F̂ + V̂, H̄ = exp(T)Ĥ exp(-T) so
        ⟨Φ_ij^ab|F|c⟩ = 0
        ⟨Φ_ij^ab|[[F, T], T]|c⟩ = 0
        ⟨Φ_ij^ab|̂V|c⟩ = ⟨ab|v̂|ij⟩
        ⟨Φ_ij^ab|[F,T]|c⟩ = P(ij)Σ_k f_i^k t_jk^ab
                          - P(ab)Σ_c f_c^a t_ij^bc  (***)
        ⟨Φ_ij^ab|[V,T]|c⟩ = ½Σ_kl⟨kl|v̂|ij⟩t_kl^ab
                          + P(ab)P(ij)Σ_kc⟨ak|v̂|ic⟩t_jk^bc
                          + ½Σ_cd⟨ab|v̂|cd⟩t_ij^cd
        ½⟨Φ_ij^ab|[[V,T], T]|c⟩ = ¼     Σ_klcd⟨kl|v|cd⟩t_kl^ab t_ij^cd
                                - ½P(ij)Σ_klcd⟨kl|v|cd⟩t_il^ab t_kj^cd
                                -  P(ab)Σ_klcd⟨kl|v|cd⟩t_ik^ac t_lj^bd
                                - ½P(ab)Σ_klcd⟨kl|v|cd⟩t_ij^ac t_kl^bd
        (***) For the iterative scheme we change these into
        ⟨Φ_ij^ab|[F,T]|c⟩** = P(ij)Σ_k≠i f_i^k t_jk^ab
                            - P(ab)Σ_c≠a f_c^a t_ij^bc  (***)

        The amplitudes is found iteratively by computing
        (t_ij^ab)^(k+1) = g((t_ij^ab)^(k))/(ϵ_i + ϵ_j - ϵ_a - ϵ_b)
        where g(...) is the modified sum shown above.
        """
