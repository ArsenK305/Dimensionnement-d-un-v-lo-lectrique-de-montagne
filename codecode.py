# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


plt.close("all")
#==============================================================================
#-------------------------------------DONNÉES_PARCOURS--------------------------
def recuperationDonneesParcours(nomFichier) :
    """
     Permet de récupérer un fichier issu du tracé GPS
     Parametre d'entree :
        nomFichier : Nom du fichier contenant les donnees du parcours
     Parametres de sortie :
        temps : Instant des points de mesure [s]
        distance : Distance parcourue [m]
        altitude : Altitude [m]
    Usage :
        temps_brut, altitude_brut, distance_brut = recuperationDonneesParcours("travail.txt")
    """    
    with open(nomFichier, 'r') as fic:
        lines = [line.strip() for line in fic.readlines()]
    del lines[0]
    temps_brut     = np.zeros(np.shape(lines))
    altitude_brut  = np.zeros(np.shape(lines))
    distance_brut  = np.zeros(np.shape(lines))
    
    for k in range(len(lines)):
        colonnes         = lines[k].split(' ')
        temps_brut[k]    = float(colonnes[2])/1000
        altitude_brut[k] = float(colonnes[0])/1000
        distance_brut[k] = float(colonnes[1])/1000

    return  altitude_brut, distance_brut, temps_brut

altitude, distance, temps = recuperationDonneesParcours("marmotte.txt")
#==============================================================================
#------------------------------DONNÉES_ECLAIREMENT-------------------------------
def recuperationDonneesEclairement(nomFichier) :
   
    with open(nomFichier, 'r') as fic:
        lines = [line.strip() for line in fic.readlines()]
    del lines[0]
    Temps     = np.zeros(np.shape(lines))
    Eclairement  = np.zeros(np.shape(lines))
       
    for k in range(len(lines)):
        colonnes    = lines[k].split(' ')
        Temps[k]    = float(colonnes[0])
        Eclairement[k] = float(colonnes[1])       
   

    return Temps, Eclairement 

temps_eclairement, Peclairement = recuperationDonneesEclairement("Eclairement_juin.txt")

#==============================================================================
#-------------------------------------DERIVATION-------------------------------
def Derivation(y,temps_ech) :

    derivation=np.zeros(len(temps_ech))
    #dg=np.zeros(len(temps))
    k=1
    for k in range(len(temps_ech)-1):
        derivation[k]=((y[k+1]-y[k-1])/(temps_ech[k+1]-temps_ech[k-1])) #Derivée à droite
    
    derivation[0]=((y[1]-y[0])/(temps_ech[1]-temps_ech[0]))
    derivation[-1]=((y[-1]-y[-2])/(temps_ech[-1]-temps_ech[-2]))

    return derivation
#==============================================================================
#------------------------------INTEGRATION-------------------------------------
def Integration(y,temps_ech): 
    """
    Intégration d'un signal 

    Parameters
    ----------
    y : signal à intégrer 
    temps_ech : temps echantilloné de ntre signal

    Returns intégrale du signal
    -------
    None.

    """
    integration = 0
    deltaT = temps_ech[1]-temps_ech[0]
    for i in range(len(y)): 
        integration += y[i]*deltaT
    return integration


def Integration2(y,temps_ech): 
    """
    

    Parameters
    ----------
    y : TYPE
        DESCRIPTION.
    temps_ech : TYPE
        DESCRIPTION.

    Returns
    -------
    integration2 : TYPE
        DESCRIPTION.

    """
    integration2 = []
    deltaT = temps_ech[1]-temps_ech[0]
    for i in range(len(y)):
        if i==0:
            integration2.append(y[i]*deltaT)
        if i>0:    
            integration2.append(y[i]*deltaT+integration2[i-1])
    return integration2

def Integration_inst(y,temps_ech): 
    """
    Intégration d'un signal 

    Parameters
    ----------
    y : signal à intégrer 
    temps_ech : temps echantilloné de ntre signal

    Returns intégrale totale du signal
    -------
    None.

    """
    integration2 = []
    deltaT = temps_ech[1]-temps_ech[0]
    for i in range(len(y)):
        if i==0:
            integration2.append(y[i]*deltaT)
        if i>0:    
            integration2.append(y[i]*deltaT)
    return integration2
#==============================================================================
#-------------------------------------POSITIVE---------------------------------
def Positive(y,temps_ech):
    """
    

    Parameters
    ----------
    y : TYPE
        DESCRIPTION.

    Returns
    -------
    pos : TYPE
        DESCRIPTION.

    """
    pos = []
    for i in range(len(y)):
        if y[i]>0 :
            pos.append(y[i])
        if y[i]<= 0: 
            pos.append(0)
    return pos
        
#==============================================================================
#-------------------------------------GLISSANTE-------------------------------
def glissante(x,Nmoy):
    # res = []
    # for i in range(1,len(y)-2):
    #     res[i] =((y[i-1] + y[i] + y[i+1])/3)
    #     res[0] = ((y[0] + y[1])/2)
    #     res[-1] = ((y[-2] + y[-1])/2)
    porte = np.ones(2*Nmoy+1)/(2*Nmoy+1)
    Nb    = len(x) 
    foo   = np.concatenate((x,np.ones(2*Nmoy)*x[Nb-1]))-x[0]   # on complete le signal par 2Nmoy fois la  derniere valeur puis on enleve la valeur initiale
    s     = np.convolve(porte,foo)+x[0]
    return s[Nmoy:Nb+Nmoy]

#==============================================================================
#-----------------------------------VARIABLES-PARCOURS-------------------------
ptsEch = 500
temps_ech=np.linspace(min(temps),max(temps),ptsEch)  # Echantillonage du temps à 500pts
tmin = np.amin(temps_ech) # en secondes 
tmax = np.amax(temps_ech) # en secondes 
deltaT = (temps_ech[1]-temps_ech[0])
altitude_ech = np.interp(temps_ech,temps,altitude) 
distance_ech = np.interp(temps_ech,temps,distance)
altitude_filtre = glissante(altitude_ech,2)   #filtrage avec une force de 2
distance_filtre = glissante(distance_ech,2)   #filtrage avec une force de 2
Vv = Derivation(altitude_filtre,temps_ech)
Vh = Derivation(distance_filtre,temps_ech)
Vlin =np.sqrt((Vv**2)+(Vh**2)) 

#---------------------------------VARIABLES-ECLIAREMENT------------------------
temps_eclairement_ech=np.linspace(min(temps_eclairement),max(temps_eclairement),500) #echantilloné à 200pts
temps_eclairement_ech_final=np.linspace(tmin,tmax,500) #Réechantillonnage pour le parcours 
Peclairement_ech=np.interp(temps_eclairement_ech_final,temps_eclairement,Peclairement)
deltaT_eclairement = (temps_eclairement_ech_final[1]-temps_eclairement_ech_final[0])
#==============================================================================
#-----------------------------------TRACE PARCOURS DE BASE---------------------
# plt.figure()
# plt.plot(distance/1000,altitude)
# plt.title('Altitude / Distance Marmotte')
# plt.xlabel('Distance [km]')
# plt.ylabel('Altitude [m]')
# plt.grid(True)

# plt.figure()
# plt.plot(temps/3600,altitude)
# plt.title('Altitude / Temps Marmotte')
# plt.xlabel('Temps [h]')
# plt.ylabel('Altitude [m]')
# plt.grid(True)
#==============================================================================
#-----------------------------------TRACE FILTRE-------------------------------
plt.figure()
plt.plot(temps_ech/3600,altitude_filtre)
plt.grid()
plt.title("Altitude filtrée du circuit en fonction du temps")
plt.xlabel('Temps [h]')
plt.ylabel('Altitude filtre [m]')
plt.show()

plt.figure()
plt.plot(temps_ech/3600,Vlin)
plt.grid()
plt.title("Vitesse du velo en fonction du temps")
plt.xlabel('Temps [h]')
plt.ylabel('Vitesse du velo [m/s]')
plt.show()

#==============================================================================
#-----------------------------------PUISSANCE TEMPS REEL-----------------------
m_cycliste = 80
m_velo = 11
m_panneau = 2
m_moteur= 5
m_batterie= 5
m = m_cycliste+m_velo+m_moteur+m_batterie+m_panneau                         # Poids en kilos
g = 9.81                                           # Constante de gravitation
f = 0.015                                          # coeff force de frottement
Scx = 0.23                                         # coeff de force aerodynamqiue
a = Derivation(Vlin,temps_ech)                    # Acceleration a chaque point
pente = Derivation (altitude_filtre,distance_filtre)     # Pente en tout point
alpha = np.arctan(pente)                               # angle de la pente

Fres = m*g*f + Scx*Vlin**2            # Force de resistance
Fvelo = m*a + Fres + m*g*alpha        # Force du velo
Pvelo = Fvelo * Vlin                  # Puissance en Watt

# plt.figure()
# plt.plot(temps_ech/3600,Pvelo)
# plt.grid()
# plt.title("Puissance instantanée")
# plt.xlabel('Temps [h]')
# plt.ylabel('Puissance [W]')
# plt.show()

#==============================================================================
#-----------------------------------ENERGIE TEMPS REEL-------------------------
E_meca_tr = (Pvelo*(deltaT/3600))
E_meca_tr_positive = Positive(E_meca_tr,temps_ech)

# plt.figure()
# plt.plot(temps_ech/3600,E_meca_tr) 
# plt.grid()
# plt.title ('Energie instantanée')
# plt.xlabel('Temps en (en h)')
# plt.ylabel('Energie(en Wh)')
# plt.legend()

# plt.figure()
# plt.plot(temps_ech/3600,E_meca_tr_positive) 
# plt.grid()
# plt.title ('Energie instantanée positive')
# plt.xlabel('Temps en (en h)')
# plt.ylabel('Energie(en Wh)')
# plt.legend()
#==============================================================================
#-----------------------------------ENERGIE TOTALE-----------------------------
E_meca_totale = Integration(Pvelo, temps_ech/3600) #Energie totale avec puissances négatives 
E_meca_totale2 = Integration2(Pvelo, temps_ech/3600) #Energie totale pour le tracé
E_meca_positive_totale = Integration2(Positive(Pvelo,temps_ech/3600),temps_ech/3600) #Energie à fournir pour le parcours 

# plt.figure()
# plt.plot(temps_ech/3600,E_meca_totale2) 
# plt.grid()
# plt.title ('Energie mécanique totale')
# plt.xlabel('Temps en (en h)')
# plt.ylabel('Energie mécanique(en Wh)')
# plt.legend()

plt.figure()
plt.plot(temps_ech/3600,E_meca_positive_totale) 
plt.grid()
plt.title ('Energie mécanique nécessaire')
plt.xlabel('Temps en (en h)')
plt.ylabel('Energie mécanique(en Wh)')
plt.legend()

#==============================================================================
#--------------------------------ECLAIREMENT-----------------------------------
plt.figure()
plt.plot(temps_eclairement_ech_final/3600,Peclairement_ech,'r') 
plt.grid()
plt.title ('Eclairement sur le parcours')
plt.xlabel('Temps en (en h)')
plt.ylabel('Puissance (en W)')

# En energie instantanée 
E_eclairement = Peclairement_ech*(deltaT_eclairement/3600)

plt.figure()
plt.plot(temps_eclairement_ech_final/3600,E_eclairement) 
plt.grid()
plt.title ('Energie instantanée éclairement')
plt.xlabel('Temps en (en h)')
plt.ylabel('Energie éclairement(en Wh)')
plt.legend()

# Au total 
E_eclairement_totale = Integration(Peclairement_ech,temps_eclairement_ech_final/3600) #Energie totale
E_eclairement_totale2 = Integration2(Peclairement_ech,temps_eclairement_ech_final/3600) #en liste pouur le tracé 

plt.figure()
plt.plot(temps_eclairement_ech_final/3600,E_eclairement_totale2) 
plt.grid()
plt.title ('Energie totale éclairement')
plt.xlabel('Temps en (en h)')
plt.ylabel('Energie éclairement(en Wh)')
plt.legend()
#==============================================================================
#--------------------------------ASSISTANCE & RECUPERATION---------------------
taux_assistance=0.5
h_mot=0.86
h_variateur=0.8

Pmot0=0
Pmot=[]
Precup=[]
for i in range(len(Pvelo)):
        if Pvelo[i]>0:
            Precup.append(0)
            if Vlin[i]<=6.9 and Vv[i]>=0:
                Pmot.append(Pvelo[i]*taux_assistance)#Assistance aide
            else:
                Pmot.append(Pmot0)
                
        else:
            Precup.append(Pvelo[i]*h_mot*h_variateur)
            Pmot.append(Pmot0)

# Pmot et Precup
plt.figure()
plt.grid()
plt.plot(temps_ech/3600,Pmot,'g') 
plt.plot(temps_ech/3600,Precup,'r') 
plt.plot(temps_ech/3600,Pvelo,'b') 
plt.title ('Puissance moteur et recuperation')
plt.xlabel('Temps en (en h)')
plt.ylabel('Puissances(en W)')
plt.legend()
        
Energie_mot=Integration_inst(Pmot,temps_ech/3600)
#==============================================================================
#--------------------------------MOTEUR----------------------------------------
#Variables
Pmot_totale=Pvelo*taux_assistance
rRoue = (622/2)/1000 #basé sur la norme ETRTO (622mm de diamètre de roue) en m
omega = Vlin/rRoue
cMot = Pmot_totale/omega

#Couple en fonction de la vitesse 
plt.figure()
plt.plot(omega,cMot,'xg',label='Couple')
plt.grid()
plt.subplot().add_patch(patches.Rectangle((0,-24.75),max(omega),43.25,edgecolor = 'red',fill=False))
plt.title ('Couple Moteur en fonction de la vitesse angulaire')
plt.xlabel('VitesseAngulaire (rad.s-1)')
plt.ylabel('Couple (en N.m)')
plt.legend()

Pmax_mot= abs(-24.75*(max(omega))) #On prends la valeur minimale pour la récupération en descente (Grandeur dimensionnante du moteur)

#Puissance électrique du moteur
Pelec_mot = []
for i in range (len(Pmot)): 
    Pelec_mot.append(Pmot[i]/h_mot)
    
# plt.figure()
# plt.grid()
# plt.plot(temps_ech/3600,Pelec_mot,'g') 
# plt.plot(temps_ech/3600,Precup,'r') 
# plt.title ('Puissance électrique du moteur et recup')
# plt.xlabel('Temps en (en h)')
# plt.ylabel('Puissances')
# plt.legend()

#Energie électrique moteur
Eelec_mot = Integration_inst(Pelec_mot,temps_ech/3600)  #Energie élctrique nécessaire pour le moteur
Eelec_recup = Integration_inst(Precup,temps_ech/3600)

Eelec_recup_pos=[]
for i in range(len(Eelec_recup)):
    Eelec_recup_pos.append(-1*Eelec_recup[i])
    
# plt.figure()
# plt.grid()
# plt.plot(temps_ech/3600,Eelec_mot,'r') 
# plt.plot(temps_ech/3600,Eelec_recup,'g') 
# plt.title ('Energie électrique du moteur et recupération')
# plt.xlabel('Temps en (en h)')
# plt.ylabel('Energies Wh')
# plt.legend()

######################## Adaptation avec un réducteur #########################
omegaMoteur= 2800  #Moteur à 2000 tr/min
omegaVelo = 335      #Vitesse angulaire maximale en tr/min
n = omegaMoteur/omegaVelo #rapport de réduction

cMot_Red= cMot/n
omega_Red= omega*n

# plt.figure()
# plt.grid()
# plt.title ('Evolution du couple en fonction de la vitesse Angulaire')
# plt.plot(omega_Red,cMot_Red,'xb')
# plt.subplot().add_patch(patches.Rectangle((0,-2.95),max(omega_Red),5.6,edgecolor = 'red',fill=False))
# plt.xlabel('VitesseAng (rad.s-1)')
# plt.ylabel('C (en N.m)')
# plt.legend()

Pmax_mot_Red= abs(-2.95*(max(omega_Red)))
#==============================================================================
#--------------------------------PANNEAU---------------------------------------
#Puissance en instantané
s_panneau = 0.5
n_panneau = 0.16
p_panneau = Peclairement_ech*s_panneau*n_panneau
rendement_charge = 0.9
P_chargeur = p_panneau*rendement_charge

# plt.figure()  
# plt.grid()  
# plt.plot(temps_eclairement_ech_final/3600,p_panneau) # tracé de la puissance du panneau en fonction du temps
# plt.title ('Puissance Panneau')
# plt.xlabel('Temps en (en h)')
# plt.ylabel('Puissance (en W)')
# plt.legend()

#Energie en instantané
E_panneau = p_panneau*(deltaT_eclairement/3600)

plt.figure()
plt.plot(temps_ech/3600,E_panneau) 
plt.grid()
plt.title ('Energie instantanée du panneau')
plt.xlabel('Temps en (en h)')
plt.ylabel('Energie panneau(en Wh)')
plt.legend()

#Energie totale
E_panneau_totale = Integration(p_panneau,temps_eclairement_ech_final/3600)
E_panneau_totale2 = Integration2(p_panneau,temps_eclairement_ech_final/3600)

plt.figure()
plt.plot(temps_eclairement_ech_final/3600,E_panneau_totale2) 
plt.grid()
plt.title ('Energie totale du panneau')
plt.xlabel('Temps en (en h)')
plt.ylabel('Energie du panneau(en Wh)')
plt.legend()

# Energie chargeur
E_chargeur = Integration_inst(P_chargeur,temps_eclairement_ech_final/3600)
E_chargeur1 =[]
for i in range(500):
    E_chargeur1.append(0)
    
E_chargeur1 = Integration2(P_chargeur,temps_eclairement_ech_final/3600)
    
plt.figure()
plt.plot(temps_eclairement_ech_final/3600,E_chargeur1) 
plt.grid()
plt.title ('Energie du chargeur')
plt.xlabel('Temps en (en h)')
plt.ylabel('Energie du chargeur(en Wh)')
plt.legend()

#==============================================================================
#--------------------------------Batterie--------------------------------------
rendement_bat=0.97 #Choix bat. au lithium ion rendement est proche de 100%
Energie_bat_3kg=600#Wh
Energie_bat_nominal_3kg=Energie_bat_3kg*rendement_bat#Wh- pour 3kg
Energie_bat_high=0.85*Energie_bat_nominal_3kg #limit high(seuil 80% pour pas endommager la batterie)
Energie_bat_low=0.2*Energie_bat_nominal_3kg   #limit low(seuil 20% pour pas endommager la batterie)
Energie_batterie_charge=Energie_bat_high #la bat pleinement chargé 
Energie_batterie_final=[]
Energie_batterie_final.append(Energie_batterie_charge)


for i in range(len(Energie_mot)):

    if Energie_batterie_final[i]<Energie_bat_high:                                                              #1.0
        #recuperer l'energie dans la bat
        if Eelec_mot[i]>0 : #SI il nous FAUT utiliser l'assistance
            #Si on PEUT utiliser l'assistance 
            if Energie_batterie_final[i]+E_chargeur[i]-Eelec_mot[i]+Eelec_recup_pos[i]<Energie_bat_high:        #1.0.1        #si on peut decharger et charger(panneau) au meme temps la batterie(total<80%)
                if Energie_batterie_final[i]+E_chargeur[i]+Eelec_recup_pos[i]-Eelec_mot[i]>Energie_bat_low:     #2.1           # si on ne depasse pas le minumum de charge(20%)
                    myvalue=Energie_batterie_final[i]-Eelec_mot[i]+E_chargeur[i]+Eelec_recup_pos[i] 
                    Energie_batterie_final.append(myvalue)#bat_final[i]=bat_final[i-1]+energie chargeur+energie recupere -energie moteur
                elif Energie_batterie_final[i]+E_chargeur[i]-Eelec_mot[i]+Eelec_recup_pos[i]<Energie_bat_low:   #2.0
                        myvalue=Energie_batterie_final[i]+E_chargeur[i] +Eelec_recup_pos[i]
                        Energie_batterie_final.append(myvalue)#bat_final[i]=bat_final[i-1]+energie chargeur+energie recupere 
            elif Energie_batterie_final[i]+Eelec_recup_pos[i] +E_chargeur[i]-Eelec_mot[i]>Energie_bat_high:     #1.1         # on ne PEUT PAS charger la bat(total>80% sinon)
                if Energie_batterie_final[i]+E_chargeur[i]-Eelec_mot[i]>Energie_bat_high:                       #1.1.1
                    myvalue=Energie_batterie_final[i]-Eelec_mot[i]
                    Energie_batterie_final.append(myvalue)#bat_final[i+1]=bat_final[i]-energie moteur
                elif  Energie_batterie_final[i]+E_chargeur[i]-Eelec_mot[i]<Energie_bat_high:                    #1.0.2
                    if Energie_batterie_final[i]+E_chargeur[i]-Eelec_mot[i]<Energie_bat_low:                    #2.0.1
                        myvalue=Energie_batterie_final[i]+E_chargeur[i]
                        Energie_batterie_final.append(myvalue)#bat_final[i+1]=bat_final[i]+energie du panneau
                    else:                                                                                       #3.0
                        myvalue=Energie_batterie_final[i]-Eelec_mot[i]+E_chargeur[i]
                        Energie_batterie_final.append(myvalue)#bat_final[i+1]=bat_final[i]-energie moteur+energie du panneau
                elif Energie_batterie_final[i]+Eelec_recup_pos[i]-Eelec_mot[i]>Energie_bat_high:                #1.1.2             # on ne PEUT PAS charger la bat(total>80% sinon)
                    myvalue=Energie_batterie_final[i]-Eelec_mot[i]
                    Energie_batterie_final.append(myvalue)#bat_final[i+1]=bat_final[i]-energie moteur  
                elif  Energie_batterie_final[i]+Eelec_recup_pos[i]-Eelec_mot[i]<Energie_bat_high:               #1.0.2
                    if Energie_batterie_final[i]+Eelec_recup_pos[i]-Eelec_mot[i]<Energie_bat_low:               #2.0.2
                        myvalue=Energie_batterie_final[i]+Eelec_recup_pos[i]
                        Energie_batterie_final.append(myvalue)#bat_final[i+1]=bat_final[i]+energie recupere
                    else:                                                                                       #3.1
                        myvalue=Energie_batterie_final[i]-Eelec_mot[i]+Eelec_recup_pos[i]
                        Energie_batterie_final.append(myvalue)#bat_final[i+1]=bat_final[i]-energie moteur+energie recupere
            
                
        elif Eelec_mot[i]==0 :   
            #Si on ne PEUT PAS utiliser l'assistance(cas de chargement)
            if Energie_batterie_final[i]+E_chargeur[i]+Eelec_recup_pos[i]<Energie_bat_high:                     #1.0.3
                myvalue=Energie_batterie_final[i]+E_chargeur[i]+Eelec_recup_pos[i]
                Energie_batterie_final.append(myvalue)
            elif Energie_batterie_final[i]+E_chargeur[i]<Energie_bat_high:                                      #1.0.4
                myvalue=Energie_batterie_final[i]+E_chargeur[i]
                Energie_batterie_final.append(myvalue)
            elif Energie_batterie_final[i]+Eelec_recup_pos[i]<Energie_bat_high:                                 #1.0.5
                myvalue=Energie_batterie_final[i]+Eelec_recup_pos[i]
                Energie_batterie_final.append(myvalue)
            else:                                                                                               #3.2
                myvalue=Energie_batterie_final[i]
                Energie_batterie_final.append(myvalue)
    else:                                                                              #3.3
        #On peut la que decharger
        myvalue=Energie_batterie_final[i]-Eelec_mot[i]
        Energie_batterie_final.append(myvalue)
        
        
Energie_batterie_final.pop()

#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================
rendement_bat=0.97 #Choix bat. au lithium ion rendement est proche de 100%
Energie_bat_5kg=1000#Wh
Energie_bat_nominal_5kg=Energie_bat_5kg*rendement_bat#Wh- pour 5kg 
Energie_bat_high1=0.85*Energie_bat_nominal_5kg #limit high(seuil 80% pour pas endommager la batterie)
Energie_bat_low1=0.2*Energie_bat_nominal_5kg   #limit low(seuil 20% pour pas endommager la batterie)
Energie_batterie_charge1=Energie_bat_high1 #la bat pleinement chargé 
Energie_batterie_final1=[]
Energie_batterie_final1.append(Energie_batterie_charge1)
for i in range(len(Energie_mot)):

    if Energie_batterie_final1[i]<Energie_bat_high1:
        #recuperer l'energie dans la bat
        if Eelec_mot[i]>0 : #SI il nous FAUT utiliser l'assistance
            #Si on PEUT utiliser l'assistance 
            if Energie_batterie_final1[i]+E_chargeur1[i]-Eelec_mot[i]+Eelec_recup_pos[i]<Energie_bat_high1:  #si on peut decharger et charger(panneau) au meme temps la batterie(total<80%)
                if Energie_batterie_final1[i]+E_chargeur1[i]+Eelec_recup_pos[i]-Eelec_mot[i]>Energie_bat_low1:# si on ne depasse pas le minumum de charge(20%)
                    myvalue=Energie_batterie_final1[i]-Eelec_mot[i]+E_chargeur1[i]+Eelec_recup_pos[i] 
                    Energie_batterie_final1.append(myvalue)#bat_final[i]=bat_final[i-1]+energie chargeur+energie recupere -energie moteur
                elif Energie_batterie_final1[i]+E_chargeur1[i]-Eelec_mot[i]+Eelec_recup_pos[i]<Energie_bat_low1:
                        myvalue=Energie_batterie_final1[i]+E_chargeur1[i] +Eelec_recup_pos[i]
                        Energie_batterie_final1.append(myvalue)#bat_final[i]=bat_final[i-1]+energie chargeur+energie recupere 
            elif Energie_batterie_final1[i]+Eelec_recup_pos[i] +E_chargeur1[i]-Eelec_mot[i]>Energie_bat_high1:# on ne PEUT PAS charger la bat(total>80% sinon)
                if Energie_batterie_final1[i]+E_chargeur1[i]-Eelec_mot[i]>Energie_bat_high1:
                    myvalue=Energie_batterie_final1[i]-Eelec_mot[i]
                    Energie_batterie_final1.append(myvalue)#bat_final[i+1]=bat_final[i]-energie moteur
                elif  Energie_batterie_final1[i]+E_chargeur1[i]-Eelec_mot[i]<Energie_bat_high1:
                    if Energie_batterie_final1[i]+E_chargeur1[i]-Eelec_mot[i]<Energie_bat_low1:
                        myvalue=Energie_batterie_final1[i]+E_chargeur1[i]
                        Energie_batterie_final1.append(myvalue)#bat_final[i+1]=bat_final[i]+energie du panneau
                    else:
                        myvalue=Energie_batterie_final1[i]-Eelec_mot[i]+E_chargeur1[i]
                        Energie_batterie_final1.append(myvalue)#bat_final[i+1]=bat_final[i]-energie moteur+energie du panneau
                elif Energie_batterie_final1[i]+Eelec_recup_pos[i]-Eelec_mot[i]>Energie_bat_high1:# on ne PEUT PAS charger la bat(total>80% sinon)
                    myvalue=Energie_batterie_final1[i]-Eelec_mot[i]
                    Energie_batterie_final1.append(myvalue)#bat_final[i+1]=bat_final[i]-energie moteur  
                elif  Energie_batterie_final1[i]+Eelec_recup_pos[i]-Eelec_mot[i]<Energie_bat_high1:
                    if Energie_batterie_final1[i]+Eelec_recup_pos[i]-Eelec_mot[i]<Energie_bat_low1:
                        myvalue=Energie_batterie_final1[i]+Eelec_recup_pos[i]
                        Energie_batterie_final1.append(myvalue)#bat_final[i+1]=bat_final[i]+energie recupere
                    else:
                        myvalue=Energie_batterie_final1[i]-Eelec_mot[i]+Eelec_recup_pos[i]
                        Energie_batterie_final1.append(myvalue)#bat_final[i+1]=bat_final[i]-energie moteur+energie recupere
            
                
        elif Eelec_mot[i]==0 :   
            #Si on ne PEUT PAS utiliser l'assistance(cas de chargement)
            if Energie_batterie_final1[i]+E_chargeur1[i]+Eelec_recup_pos[i]<Energie_bat_high1:
                myvalue=Energie_batterie_final1[i]+E_chargeur1[i]+Eelec_recup_pos[i]
                Energie_batterie_final1.append(myvalue)
            elif Energie_batterie_final1[i]+E_chargeur1[i]<Energie_bat_high1:
                myvalue=Energie_batterie_final1[i]+E_chargeur1[i]
                Energie_batterie_final1.append(myvalue)
            elif Energie_batterie_final1[i]+Eelec_recup_pos[i]<Energie_bat_high1:
                myvalue=Energie_batterie_final1[i]+Eelec_recup_pos[i]
                Energie_batterie_final1.append(myvalue)
            else:
                myvalue=Energie_batterie_final1[i]
                Energie_batterie_final1.append(myvalue)
    else:
        #On peut la que decharger
        myvalue=Energie_batterie_final1[i]-Eelec_mot[i]
        Energie_batterie_final1.append(myvalue)
Energie_batterie_final1.pop()

plt.figure()
plt.plot(temps_ech/3600,Energie_batterie_final,"b") 
plt.plot(temps_ech/3600,Energie_batterie_final1,"r")
plt.grid()
plt.title ('Energie totale de la batterie')
plt.xlabel('Temps en (en h)')
plt.ylabel('Energie de la batterie(en Wh)')
plt.legend()
