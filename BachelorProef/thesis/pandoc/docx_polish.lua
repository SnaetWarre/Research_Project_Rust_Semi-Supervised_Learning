--- Pandoc Lua filter: nicer .docx for thesis builds
--- - Page break before every level-1 heading except the first front-matter heading.
--- - Default figure width so SVGs/charts do not spill past margins (override per image with {width=...}).
--- - Keep only numbered level-1 and level-2 headings in the generated table of contents.

local PAGE_BREAK = pandoc.RawBlock("openxml", "<w:p><w:r><w:br w:type=\"page\"/></w:r></w:p>")

local h1_count = 0

local function heading_text(el)
  return pandoc.utils.stringify(el.content)
end

local function is_toc_heading(el)
  if el.level > 2 then
    return false
  end
  return heading_text(el):match("^%d+%.?%d*%s+") ~= nil
end

function Header(el)
  if not is_toc_heading(el) then
    el.classes:insert("unlisted")
  end

  if el.level == 1 then
    h1_count = h1_count + 1
    if h1_count > 1 then
      return { PAGE_BREAK, el }
    end
  end
  return el
end

function Image(img)
  local attrs = img.attr.attributes
  local w = attrs.width
  if w == nil or w == "" then
    attrs.width = "5.9in"
  end
  -- Centre the image paragraph in Word output
  local center_open = pandoc.RawInline("openxml",
    '<w:pPr><w:jc w:val="center"/></w:pPr>')
  return { center_open, img }
end
