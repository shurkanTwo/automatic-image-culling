local LrApplication = import 'LrApplication'
local LrDialogs = import 'LrDialogs'
local LrFileUtils = import 'LrFileUtils'
local LrFunctionContext = import 'LrFunctionContext'
local LrPathUtils = import 'LrPathUtils'
local LrProgressScope = import 'LrProgressScope'
local LrTasks = import 'LrTasks'

local json = dofile(LrPathUtils.child(_PLUGIN.path, 'json.lua'))
local Manifest = dofile(LrPathUtils.child(_PLUGIN.path, 'Manifest.lua'))

local function loadSelection(path)
    local contents = LrFileUtils.readFile(path)
    if not contents then error('The selection file could not be read.') end
    if #contents > 50 * 1024 * 1024 then error('The selection file is too large.') end
    return Manifest.validate(json.decode(contents))
end

local function writeCatalog(catalog, name, callback)
    local status = catalog:withWriteAccessDo(name, callback, { timeout = 30 })
    if status ~= 'executed' then
        error('Lightroom is busy and could not complete the import. Try again when other catalog operations have finished.')
    end
end

local function importSelection(context)
    local selected = LrDialogs.runOpenPanel {
        title = 'Import a Photo Select shortlist', prompt = 'Review import',
        canChooseFiles = true, canChooseDirectories = false,
        allowsMultipleSelection = false, fileTypes = { 'json' },
    }
    if not selected or not selected[1] then return end

    local payload = loadSelection(selected[1])
    local catalog = LrApplication.activeCatalog()
    local progress = LrProgressScope {
        title = 'Matching Photo Select photographs', functionContext = context,
    }
    local matched, seen, missing = {}, {}, 0
    local ratings, favorites, discards, tags, members = 0, 0, 0, 0, 0
    for index, entry in ipairs(payload.photos) do
        if progress:isCanceled() then return end
        local photo = catalog:findPhotoByPath(entry.path, false)
        if photo then
            if not seen[photo.localIdentifier] then
                matched[#matched + 1] = { photo = photo, entry = entry }
                seen[photo.localIdentifier] = true
                if entry.rating ~= nil then ratings = ratings + 1 end
                local flag = Manifest.catalogFlag(payload, entry)
                if flag == 'pick' then favorites = favorites + 1 end
                if flag == 'reject' then discards = discards + 1 end
                if Manifest.addToCollection(payload, entry) then members = members + 1 end
                if #entry.tags > 0 then tags = tags + 1 end
            end
        else
            missing = missing + 1
        end
        progress:setPortionComplete(index, #payload.photos)
        if index % 50 == 0 then LrTasks.yield() end
    end
    progress:done()
    if #matched == 0 then
        LrDialogs.message('No matching photographs',
            'Import these originals into Lightroom first, then try again. File paths must match the paths in the exported selection.', 'info')
        return
    end

    local summary = string.format(
        'Add %d photographs to Photo Select / %s.\n\nApply %d Pick flags, %d Reject flags, %d explicitly changed star ratings, and tags on %d photographs.\n%d photographs were not found in this catalog and will be skipped.\n\nReject flags do not delete photographs. Existing keywords and Develop settings are preserved.',
        members, Manifest.collectionName(payload), favorites, discards, ratings, tags, missing)
    if LrDialogs.confirm('Import this shortlist?', summary, 'Import', 'Cancel') ~= 'ok' then return end

    local collection, keywords
    keywords = {}
    -- New SDK objects can only be used after the write gate that creates them returns.
    writeCatalog(catalog, 'Prepare Photo Select shortlist', function()
        if members > 0 then
            local parent = catalog:createCollectionSet('Photo Select', nil, true)
            collection = catalog:createCollection(Manifest.collectionName(payload), parent, true)
            if not collection then error('Lightroom could not create the selection collection.') end
        end
        for _, item in ipairs(matched) do
            for _, tag in ipairs(item.entry.tags) do
                if not keywords[tag] then
                    keywords[tag] = catalog:createKeyword(tag, {}, true, nil, true)
                    if not keywords[tag] then error('Lightroom could not create the keyword: ' .. tag) end
                end
            end
        end
    end)

    writeCatalog(catalog, 'Import Photo Select shortlist', function()
        local photos = {}
        for _, item in ipairs(matched) do
            local photo, entry = item.photo, item.entry
            if entry.rating ~= nil then photo:setRawMetadata('rating', entry.rating) end
            local flag = Manifest.catalogFlag(payload, entry)
            if flag == 'pick' then photo:setRawMetadata('pickStatus', 1)
            elseif flag == 'reject' then photo:setRawMetadata('pickStatus', -1) end
            for _, tag in ipairs(entry.tags) do
                photo:addKeyword(keywords[tag])
            end
            if Manifest.addToCollection(payload, entry) then photos[#photos + 1] = photo end
        end
        if collection then collection:addPhotos(photos) end
    end)
    LrDialogs.message('Shortlist imported', string.format(
        '%d photographs added to Photo Select / %s.\n%d Pick flags and %d Reject flags applied.\n%d unmatched photographs skipped. You can undo the metadata import using Lightroom\'s Undo command.',
        members, Manifest.collectionName(payload), favorites, discards, missing), 'info')
end

LrTasks.startAsyncTask(function()
    local ok, message = LrTasks.pcall(function()
        LrFunctionContext.callWithContext('Photo Select import', importSelection)
    end)
    if not ok then LrDialogs.message('Photo Select import failed', tostring(message), 'critical') end
end)
