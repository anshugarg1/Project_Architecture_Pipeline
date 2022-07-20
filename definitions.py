night_folder_list = ['2014-12-17-18-18-43', '2014-12-10-18-10-50', '2014-11-14-16-34-33']
alternate_route_folder_list = ['2014-05-06-12-54-54']
snow_folder_list = ['2015-02-03-08-45-10']
rain_folder_list = ['2014-11-21-16-07-03', '2014-11-25-09-18-32', '2015-10-29-12-18-17']
sun_folder_list = ['2015-11-12-11-22-05', '2014-07-14-15-42-55', '2015-04-21-12-10-38', '2015-08-12-15-04-18','2014-07-14-15-16-36']

mode_folder_mapping = {
    "train_sun":sun_folder_list,
    "test_alternate_route": alternate_route_folder_list,
    "test_night":night_folder_list,
    "test_snow":snow_folder_list,
    "test_rain":rain_folder_list
}

mean = [0.2528, 0.2588, 0.3199]
std = [0.9815, 0.9927, 0.9684]