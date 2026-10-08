local Manifest = {}

local function validText(value, limit)
    return type(value) == 'string' and #value > 0 and #value <= limit
end

function Manifest.validate(payload)
    if type(payload) ~= 'table' or payload.schemaVersion ~= 1
        or payload.application ~= 'Photo Select' or type(payload.photos) ~= 'table' then
        error('Choose a selection JSON exported by Photo Select.')
    end
    if not validText(payload.projectName, 500) or not validText(payload.collectionName, 500) then
        error('The selection is missing its project or collection name.')
    end
    if #payload.photos > 100000 then
        error('This selection is too large to import in one operation.')
    end
    for index, entry in ipairs(payload.photos) do
        if type(entry) ~= 'table' or not validText(entry.path, 32768) then
            error('Photo ' .. index .. ' has an invalid file path.')
        end
        if entry.rating ~= nil and (type(entry.rating) ~= 'number'
            or entry.rating < 0 or entry.rating > 5 or entry.rating % 1 ~= 0) then
            error('Photo ' .. index .. ' has an invalid star rating.')
        end
        if entry.decision ~= 'favorite' and entry.decision ~= 'pass'
            and entry.decision ~= 'undecided' then
            error('Photo ' .. index .. ' has an invalid review decision.')
        end
        if type(entry.tags) ~= 'table' or #entry.tags > 100 then
            error('Photo ' .. index .. ' has an invalid tag list.')
        end
        for _, tag in ipairs(entry.tags) do
            if not validText(tag, 250) or tag:find('[,;|<>%c]') then
                error('A tag contains characters Lightroom does not support.')
            end
        end
    end
    return payload
end

function Manifest.collectionName(payload)
    return payload.projectName .. ' - ' .. payload.collectionName
end

return Manifest
