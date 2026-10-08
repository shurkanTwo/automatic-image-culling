local Manifest = {}

local function validText(value, limit)
    if type(value) ~= 'string' or #value == 0 then return false end
    -- Lua 5.1 counts bytes; the app's limits count Unicode characters.
    local characters = value:gsub('[\128-\191]', '')
    return #characters <= limit
end

local function validList(value, limit)
    if type(value) ~= 'table' or #value > limit then return false end
    local count = 0
    for key in pairs(value) do
        if type(key) ~= 'number' or key % 1 ~= 0 or key < 1 or key > #value then
            return false
        end
        count = count + 1
    end
    return count == #value
end

function Manifest.validate(payload)
    if type(payload) ~= 'table' or payload.schemaVersion ~= 1
        or payload.application ~= 'Photo Select' or not validList(payload.photos, 100000) then
        error('Choose a selection JSON exported by Photo Select.')
    end
    if not validText(payload.projectName, 200) or not validText(payload.collectionName, 200) then
        error('The selection is missing its project or collection name.')
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
        if not validList(entry.tags, 100) then
            error('Photo ' .. index .. ' has an invalid tag list.')
        end
        for _, tag in ipairs(entry.tags) do
            if not validText(tag, 100) or tag:find('[,;|<>%c]') then
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
