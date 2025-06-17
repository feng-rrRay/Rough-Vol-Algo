function flag = json_add(jsonFileName, structToAdd)
% JSON_ADD Adds a struct to the end of a JSON file that contains a struct or a struct array.
%   FLAG = JSON_ADD(JSONFILENAME, STRUCTTOADD) reads the JSON file and appends
%   STRUCTTOADD to it. If the file does not exist, it creates a new JSON file containing
%   a single struct. If the file exists but its contents are not a struct (or struct array),
%   the function returns 1.

    % Ensure the input is a struct.
    if ~isstruct(structToAdd)
        error('Input to append must be a struct.');
    end

    if ~isfile(jsonFileName)
        % File does not exist: create a new JSON file with structToAdd as a single struct.
        newData = structToAdd;  % 1-by-1 struct
        fp = fopen(jsonFileName, 'w');
        if fp == -1
            error('Could not open file for writing.');
        end
        fprintf(fp, '%s', jsonencode(newData, "PrettyPrint", true));
        fclose(fp);
        flag = 0;
        return;
    end

    % File exists: read and decode its content.
    fileContent = fileread(jsonFileName);
    try
        ex = jsondecode(fileContent);
    catch
        error('Error decoding the JSON file.');
    end

    % Check that the existing content is a struct or struct array.
    if ~isstruct(ex)
        flag = 1;
        fprintf('Error: The existing JSON file does not contain a struct or a struct array.\n');
        return;
    end

    % If ex is a single struct (1-by-1), convert it into a 1-by-2 struct array.
    if numel(ex) == 1
        ex = [ex, structToAdd];
    else
        ex(end+1) = structToAdd;
    end

    % Write the updated struct array back to the JSON file.
    fp = fopen(jsonFileName, 'w');
    if fp == -1
        error('Could not open file for writing.');
    end
    fprintf(fp, '%s', jsonencode(ex, "PrettyPrint", true));
    fclose(fp);
    flag = 0;
end