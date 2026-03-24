def svd(A,t):
  #cal ATA
  ATA = np.dot(A.T,A)
  
  #calc eigen value and vectors 
  ATA_eig_val, ATA_eig_vect = np.linalg.eigh(ATA)
  
  # remove the nan and negative value due to floating point error
  ATA_eig_val[ATA_eig_val < t ] = 0
  ATA_eig_val[np.isnan(ATA_eig_val)]=0
  #calculate the Sigmas as its square root of the eig value matrix
  ATA_sigmas = np.sqrt(ATA_eig_val)
  ATA_sigma_matrix = np.diag(ATA_sigmas)
  
  #now calc V , sigma inverse and A.v.sigmainv = U
  V = ATA_eig_vect
  sigma_inv = np.zeros_like(ATA_sigmas)
  mask = ATA_sigmas > t
  sigma_inv[mask] = 1 / ATA_sigmas[mask]
  U = A @ V @ np.diag(sigma_inv)
  
  return U,ATA_sigma_matrix,V.T  
