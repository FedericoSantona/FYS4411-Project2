"""
    def accept_jax(self, n_accepted , accept,  initial_positions , proposed_positions ,log_psi_current,  log_psi_proposed):

        new_positions = self.backend.zeros_like(initial_positions)
        new_logp = self.backend.zeros_like(log_psi_current)

        support1 = jnp.matmul(accept.T , proposed_positions)
        support2 = jnp.matmul(1-accept.T , initial_positions) 

        new_positions = support1 + support2

        n_accepted = jnp.sum(accept)

        new_logp = jnp.where(accept, log_psi_proposed, log_psi_current)
        
        return  new_positions, new_logp , n_accepted
    
    
    def accept_numpy (self, n_accepted , accept,  initial_positions , proposed_positions ,log_psi_current,  log_psi_proposed):

        new_positions = self.backend.zeros_like(initial_positions)
        new_logp = self.backend.zeros_like(log_psi_current)

        for i in range(initial_positions.shape[0]):
            if accept[i]:
                n_accepted += 1
                new_positions[i] = proposed_positions[i]
                new_logp[i] = log_psi_proposed[i]
            else:
                new_positions[i] = initial_positions[i]
                new_logp[i] = log_psi_current[i]
        
        return  new_positions, new_logp , n_accepted
    
        
    """




"""
        sampled_positions = []
        local_energies = []  # List to store local energies
        total_accepted = 0  # Initialize total number of accepted moves
        
        
        if self._log:
            t_range = tqdm(
                range(nsamples),
                desc="[Sampling progress]",
              #  position=0,
                leave=True,
                colour="green",
            )
        else:
            t_range = range(nsamples)

        

        for _ in range(nsamples):
            # Perform one step of the MCMC algorithm

            #print( "position BEFORE ", self.alg.state.positions)
            new_state  = self.sampler.step(total_accepted, self.logp, self.alg.state, self._seed )
            
            total_accepted = new_state.n_accepted

            self.alg.state = new_state

            #print( "position AFTER ", self.alg.state.positions)

            # Calculate the local energy
        
            E_loc = self.hamiltonian.local_energy(self.wf, new_state.positions)


            #print("this is the local energy" , self._backend,  E_loc.shape)
            
            local_energies.append(E_loc)  # Store local energy 

            # Store sampled positions and calculate acceptance rate
            sampled_positions.append(new_state.positions)
        
        # Calculate acceptance rate
        acceptance_rate = total_accepted / (nsamples*self._N)

        local_energies = self.backend.array(local_energies)


        #print("local_energies", local_energies)

        # Compute statistics of local energies
        mean_energy = self.backend.mean(local_energies)
        std_error = self.backend.std(local_energies) / self.backend.sqrt(nsamples)
        variance = self.backend.var(local_energies)



        #OBS: this should actually be returned from the sampler sample method. This is as is below just a placeholder
        # Update the sample_results dictionary
        sample_results = {
            "chain_id": None,
            "energy": mean_energy,
            "std_error": std_error,
            "variance": variance,
            "accept_rate": acceptance_rate,
            "scale": self.sampler.scale,
            "nsamples": nsamples,
        }

        """