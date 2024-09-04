function check_reward_zone_area(croped_track,track_len_cm,reward_zone_cm,reward_zone_bodge )

track_len_pix = size(croped_track,2);

pixratio = track_len_pix/track_len_cm;
reward_zone_pix = reward_zone_cm * pixratio;
rightRZ = reward_zone_cm - reward_zone_bodge;
rightRZ_pix = rightRZ * pixratio;

figure
imshow(croped_track)
hold on 
line([reward_zone_pix reward_zone_pix], [0,83])
hold on 
line([track_len_pix-rightRZ_pix track_len_pix-rightRZ_pix], [0,83])

end
