import numpy as np



pt1 = 10
phi1 = 0.5
eta1 = 1


pt2 = 13
phi2 = 0.6
eta2 = 1.1

px1=pt1*np.cos(phi1)
py1=pt1*np.sin(phi1)
pz1=pt1*np.sinh(eta1)
#costheta1=pz1*1./np.sqrt(px1*px1+py1*py1+pz1*pz1)
# print(‘Efficiency:’)
# print(pt1.shape[0]*1./p.shape[0])
px2=pt2*np.cos(phi2)
py2=pt2*np.sin(phi2)
pz2=pt2*np.sinh(eta2)
#costheta2=pz2*1./np.sqrt(px2*px2+py2*py2+pz2*pz2)
E1=np.sqrt(0.1**2+(px1*px1+py1*py1+pz1*pz1))
E2=np.sqrt(0.1**2+(px2*px2+py2*py2+pz2*pz2))
delta_phi=np.abs(phi2-phi1)
mll=np.sqrt((E1+E2)**2-(px1+px2)**2-(py1+py2)**2-(pz1+pz2)**2)

print(mll)



muon_mass=0.1 #GeV/c2
# pt1 =HLF[:, 0]
# pt2 =HLF[:, 1]
# eta1=HLF[:, 2]
# eta2=HLF[:, 3]


dphi=delta_phi
px1=pt1
px2=pt2*np.cos(dphi)

py1=np.zeros_like(pt1)
py2=pt2*np.sin(dphi)

pz1=pt1*np.sinh(eta1)
pz2=pt2*np.sinh(eta2)


E1 =np.sqrt(px1*px1+py1*py1+pz1*pz1+muon_mass*muon_mass)
E2 =np.sqrt(px2*px2+py2*py2+pz2*pz2+muon_mass*muon_mass)
px=px1+px2
py=py1+py2
pz=pz1+pz2
E =E1+E2
mll=np.sqrt(E*E-px*px-py*py-pz*pz)


print(mll)