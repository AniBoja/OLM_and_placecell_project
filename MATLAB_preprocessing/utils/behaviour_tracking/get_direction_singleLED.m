function [Frame_direction_binary] = get_direction_singleLED(cords1)

buffer = 5;
Frame_direction = zeros(size(cords1,1),1);
for i = buffer:size(cords1,1)

    x_cords = cords1(:,1);
    prev_cords = x_cords(i-(buffer-1):i-1,1);
    current_cord = x_cords(i);

    if all(prev_cords<current_cord)
        Frame_direction(i) = 0;
    elseif all(prev_cords > current_cord)
        Frame_direction(i) = 1;
    else
        Frame_direction(i) = Frame_direction(i-1);
    end

end
   

Frame_direction_binary = Frame_direction;
end