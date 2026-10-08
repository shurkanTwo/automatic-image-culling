-- A small test double for the Lightroom SDK; never opens a real catalog.
TEST_PHOTOS = {}
TEST_MESSAGES = {}
TEST_COLLECTIONS = {}
TEST_WRITES = 0
TEST_CONFIRM = 'ok'
TEST_CANCEL = false

function testPhoto(path, rating, pick)
    local photo = {
        localIdentifier = path,
        metadata = { rating = rating, pickStatus = pick },
        keywords = { existing = true },
        developSettings = 'unchanged',
    }
    function photo:setRawMetadata(key, value)
        assert(TEST_IN_WRITE, 'Metadata writes require catalog write access')
        self.metadata[key] = value
    end
    function photo:addKeyword(keyword)
        assert(TEST_IN_WRITE, 'Keyword writes require catalog write access')
        self.keywords[keyword.name] = true
    end
    TEST_PHOTOS[path] = photo
    return photo
end

local catalog = {}
function catalog:findPhotoByPath(path)
    return TEST_PHOTOS[path]
end
function catalog:withWriteAccessDo(name, callback)
    assert(not TEST_IN_WRITE, 'Write access must not be nested')
    TEST_IN_WRITE = true
    callback()
    TEST_IN_WRITE = false
    TEST_WRITES = TEST_WRITES + 1
end
function catalog:createCollectionSet(name)
    assert(TEST_IN_WRITE)
    return { name = name }
end
function catalog:createCollection(name)
    assert(TEST_IN_WRITE)
    local existing = TEST_COLLECTIONS[name]
    if existing then return existing end
    local collection = { photos = {}, createdAtWrite = TEST_WRITES }
    function collection:addPhotos(photos)
        assert(TEST_IN_WRITE)
        assert(TEST_WRITES > self.createdAtWrite, 'Commit collection creation before adding photos')
        for _, photo in ipairs(photos) do self.photos[photo.localIdentifier] = true end
    end
    TEST_COLLECTIONS[name] = collection
    return collection
end
function catalog:createKeyword(name)
    assert(TEST_IN_WRITE)
    return { name = name }
end

local modules = {
    LrApplication = { activeCatalog = function() return catalog end },
    LrDialogs = {
        runOpenPanel = function() return { 'selection.json' } end,
        confirm = function() return TEST_CONFIRM end,
        message = function(title, message, severity)
            TEST_MESSAGES[#TEST_MESSAGES + 1] = { title = title, message = message, severity = severity }
        end,
    },
    LrFileUtils = { readFile = function() return TEST_SELECTION_JSON end },
    LrFunctionContext = { callWithContext = function(name, callback) callback({}) end },
    LrPathUtils = { child = function(parent, name) return parent .. '/' .. name end },
    LrProgressScope = function()
        return {
            isCanceled = function() return TEST_CANCEL end,
            setPortionComplete = function() end,
            done = function() end,
        }
    end,
    LrTasks = {
        startAsyncTask = function(callback) callback() end,
        pcall = pcall,
        yield = function() end,
    },
}

function import(name)
    assert(modules[name], 'Unsupported SDK module: ' .. name)
    return modules[name]
end
