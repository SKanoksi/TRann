# -*- Download Thailand meteorological data -*-

#---------- Input ----------#

# - - - - - - - - - - - - - - - - - #
#          Downloaded data
# First set  = 26/06/18 to 23/10/18
# - - - - - - - - - - - - - - - - - #
# Missing    = 24-26/10/18
# Second set = 27/10/18 - 27/11/18
# - - - - - - - - - - - - - - - - - #
# Now = 27/11/18 - 30/4/19
# DD = '10'  # (yester)Day 01-31

MM = '06'    # Month 01-12
YY = '19'    # Year 00-99
CC = '20'    # Century 20-XX
To = 'D:\ShareUbuntu\WF-AI\Thai-Met data'   # Path to directory
Sat_hour = range(0,24,1)                    # Download Satellite image every ... hour
Sat_min  = ['00','10','20','30','40','50']  # For each hour download at ... minute  (No 0000)

#---------- Main -----------#

import urllib
for DD in range(11,15+1,1) :
    if DD < 10 :
        DD = '0'+str(DD)
    else:
        DD = str(DD)

    # Setellite VIS,WV,IR/ ASIA
    print("\n  Start download All Satellite images.")
    web = 'http://www.sattmet.tmd.go.th/satellite_Data_Image/'
    for AA in ['asia'] :
        for SS in ['IR','VS','WV'] :
            prefix = AA+'/'+CC+YY+MM+DD+'/'+SS+CC+YY+MM+DD
            prefix_Save = AA+'/'+SS+AA+'.'+CC+YY+'_'+MM+DD+'_'
            for HH in Sat_hour :
                if HH < 10 :
                    hr = '0'+str(HH)
                else:
                    hr = str(HH)
                for min in Sat_min:
                    name = hr+min+'.JPG'
                    try :
                        data = urllib.request.urlopen( web+prefix+name ).read()
                    except urllib.error.HTTPError :
                        print("Download : "+prefix+name+" = failed, the file may not avaliable.")
                    else :
                        file = open( To+'/Satellite/'+prefix_Save+name ,'wb')
                        file.write(data)
                        file.close()
                        print("Download : "+prefix+name+" = success")
    '''
    # Surface pressure, Temperature, Humidity, Raing 1,4,7,10,13,16,19,22
    web = 'http://www.arcims.tmd.go.th/AUTORUN/GIFMAP3hrs/'
    for Type in ['MSL','TEMP','RH','RAIN'] :
        print("\n  Start download : "+Type+".")
        for hr in range(1,24,3):
            name = Type+YY+MM+DD+str(hr)+'.gif'
            try :
                data = urllib.request.urlopen( web+Type+DD+MM+YY+str(hr)+'.gif' ).read()
            except urllib.error.HTTPError :
                print("Download : "+name+" = failed, the file may not avaliable.")
            else :
                file = open( To+'/Statistic/'+Type+'/'+name ,'wb')
                file.write(data)
                file.close()
                print("Download : "+name+" = success")

    # Rain sum24
    print("\n  Start download : Rain24hr.")
    web = 'http://www.arcims.tmd.go.th/AUTORUN/BITMAP/'
    name = 'RAIN'+YY+MM+DD+'24-SUM'+'.gif'
    try :
        data = urllib.request.urlopen( web+'RAIN'+DD+MM+YY+'.gif' ).read()
    except urllib.error.HTTPError :
        print("Download : "+name+" = failed, the file may not avaliable.")
    else :
        file = open( To+'/Statistic/RAIN/'+name ,'wb')
        file.write(data)
        file.close()
        print("Download : "+name+" = success")
    '''
    print("---------------------------------------------------------------\n")
    print("END !! download Thailand metheorogical data of "+DD+'/'+MM+'/'+YY)
