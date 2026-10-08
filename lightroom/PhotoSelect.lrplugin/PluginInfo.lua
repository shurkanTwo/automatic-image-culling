local LrView = import 'LrView'

return {
    sectionsForTopOfDialog = function()
        local factory = LrView.osFactory()
        return {
            {
                title = 'Photo Select 0.2.1',
                factory:static_text {
                    title = 'Export a shortlist in Photo Select, then choose Library > Plug-in Extras > Import Photo Select shortlist.\n\nPhotos must already be in this Lightroom catalog at the same file paths. The import adds a collection, applies explicitly changed star ratings and adds your tags. Favorites receive a Pick flag. Existing keywords and Develop settings are preserved. Passed photos are never marked for deletion.',
                    width_in_chars = 65,
                    height_in_lines = 9,
                },
            },
        }
    end,
}
