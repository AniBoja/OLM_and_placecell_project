function [Frame_direction_binary] = get_direction(cords1, cords2, LED_dist)


LEDDiff =  cords1(:,2)- cords2(:,2);
LEDDiff_smooth = smooth(LEDDiff,100);


LEDDiff_binary = LEDDiff_smooth;
LEDDiff_binary(LEDDiff_smooth>LED_dist,:)=1;
LEDDiff_binary(LEDDiff_smooth<-LED_dist,:)=0;



Frame_direction = LEDDiff_binary;
Frame_direction_binary = LEDDiff_binary;
Frame_direction_binary(Frame_direction~=1&Frame_direction~=0,:)=NaN;

end