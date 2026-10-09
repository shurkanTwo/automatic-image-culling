PHOTO SELECT LIGHTROOM CLASSIC BRIDGE - 0.2.3

1. In Lightroom Classic, open File > Plug-in Manager > Add.
2. Select this entire PhotoSelect.lrplugin folder.
3. In Photo Select, export Favorites or a purpose collection as a selection JSON.
4. In Lightroom Classic, choose Library > Plug-in Extras > Import Photo Select
   shortlist. Depending on your version, Plug-in Extras is also in the File menu.
5. Choose the JSON, review the matching-photo summary, and click Import.

The originals must already be imported into the active Lightroom catalog and
remain at their original paths. Missing photographs are skipped and counted.
The plug-in adds matched photographs to a collection in the Photo Select
collection set. Reimporting adds to the existing collection without duplicates.
It only changes star ratings explicitly assigned in Photo Select, marks
Favorites with Pick flags, and adds tags. With Include discards as Lightroom Rejects
enabled during export, Discards become actual Reject flags. Extra discards outside
your shortlist do not join its collection. Existing keywords and Develop settings
remain intact. Reject flags never delete photographs. Old schema 1 exports retain
their previous archive-only Pass behavior. Use this updated plugin for schema 2.

No originals are copied, moved, modified, or deleted. This is a one-way import;
changes made afterward in Lightroom do not synchronize back to Photo Select.

If the catalog is busy, the import reports a failure and can be retried. New
collections and keywords are prepared first; metadata and collection membership
are then applied together. A failed second step may leave an empty collection
and unused keywords, but does not partially change your photographs.

The included json.lua decoder is MIT-licensed, copyright (c) 2020 rxi.
Its license text appears at the top of json.lua.
