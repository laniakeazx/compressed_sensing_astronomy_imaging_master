
import random



#20.0 -29.8 15 0 0 0 100.0e6 -0.7 0.0 0   0   0
#19.5 -30.3 5 2 2 0 100.0e6 9.7 40.0 600 500  40
#20.5 -30.2 5 0 0 2 100.0e6 3.8 10.0 -30 -30 -30


sky_model2osm_all = ''
for i in range(1000):
    #Right Ascension
    # RA_deg_max = 16 #18
    # RA_deg_min = 20 #22
    RA_deg_max = 20.099 #18
    RA_deg_min = 20.001 #22


    RA_deg_ponit1 = 3
    RA_deg=random.uniform(RA_deg_max,RA_deg_min)
    RA_deg_result = round(RA_deg,RA_deg_ponit1)
    # print(RA_deg_result)
    #Declination

    # Dec_deg_max = -26
    # Dec_deg_min = -30
    Dec_deg_max = -30.001
    Dec_deg_min = -30.099


    Dec_deg_ponit1 = 3
    Dec_deg=random.uniform(Dec_deg_max,Dec_deg_min)
    Dec_deg_result = round(Dec_deg,Dec_deg_ponit1)
    # print(Dec_deg_result)
    #Stokes I Q U V flux
    I_Jy_max = 3.5
    I_Jy_min = 0.5
    I_Jy_ponit1 = 0
    I_Jy=random.uniform(I_Jy_max,I_Jy_min)
    I_Jy_result = round(I_Jy)
    # print(I_Jy_result)
    Q_Jy_max = 0
    Q_Jy_min = 0
    Q_Jy_ponit1 = 0
    Q_Jy=random.uniform(Q_Jy_max,Q_Jy_min)
    Q_Jy_result = round(Q_Jy)
    # print(Q_Jy_result)
    U_Jy_max = 0
    U_Jy_min = 0
    U_Jy_ponit1 = 0
    U_Jy=random.uniform(U_Jy_max,U_Jy_min)
    U_Jy_result = round(U_Jy)
    # print(U_Jy_result)
    V_Jy_max = 0
    V_Jy_min = 0
    V_Jy_ponit1 = 0
    V_Jy=random.uniform(V_Jy_max,V_Jy_min)
    V_Jy_result = round(V_Jy)
    # print(V_Jy_result)
    # Reference frequency
    frequency_Hz = 100.0e6
    # print(frequency_Hz)
    # Spectral index
    spix = -0.7
    #Rotation measure
    RM_rad_m2 = 0.0
    # Major axis FWHM
    maj_arcsec_max = 10
    maj_arcsec_min = 0
    maj_arcsec_ponit1 = 0
    maj_arcsec=random.uniform(maj_arcsec_max,maj_arcsec_min)
    maj_arcsec_result = round(maj_arcsec)
    # print(maj_arcsec_result)
    # Minor axis FWHM
    min_arcsec_max = 10
    min_arcsec_min = 0
    min_arcsec_ponit1 = 0
    min_arcsec=random.uniform(min_arcsec_max,min_arcsec_min)
    min_arcsec_result = round(min_arcsec)
    # print(min_arcsec_result)
    # Position angle
    pa_deg_max = 90
    pa_deg_min = -90
    pa_deg_ponit1 = 0
    pa_deg=random.uniform(pa_deg_max,pa_deg_min)
    pa_deg_result = round(pa_deg)
    # print(V_Jy_result)
    sky_model2osm = [RA_deg_result,Dec_deg_result,I_Jy_result,Q_Jy_result,U_Jy_result,V_Jy_result,frequency_Hz,spix,RM_rad_m2,maj_arcsec_result,min_arcsec_result,pa_deg_result]
    sky_model2osm1 = str(sky_model2osm)
    sky_model2osm1 = sky_model2osm1.replace(',',' ')
    sky_model2osm1 = sky_model2osm1.replace('[','')
    sky_model2osm1 = sky_model2osm1.replace(']','')
    with open('data.txt', 'a+', encoding='utf-8') as f:
        f.write('\n')
        for data in sky_model2osm1:
            # 添加‘\n’用于换行

            f.write(data)
        f.close()

#     sky_model2osm_all.append(sky_model2osm1,'\n')
#
# sky_model2osm_all_list = str(sky_model2osm_all).replace(',',' ')
# sky_model2osm_all_list = sky_model2osm_all_list.replace('[','')
# sky_model2osm_all_list = sky_model2osm_all_list.replace(']','')
# print(sky_model2osm_all_list)
# print(sky_model2osm_all_list[64])



