function rotateImage(src,evt,hIm,im)


% Only rotate the image when the ROI is rotated. Determine if the
% RotationAngle has changed
if evt.PreviousRotationAngle ~= evt.CurrentRotationAngle

    % Update the label to display current rotation
    src.Label = 'Double click when done';

    % Rotate the image and update the display
    im = imrotate(im,evt.CurrentRotationAngle,'nearest','crop');
    hIm.CData = im;

end


end
