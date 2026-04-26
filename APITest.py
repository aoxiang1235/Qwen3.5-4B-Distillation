
24
curl --max-time 120 -sS http://127.0.0.1:8012/generate \
  -H "Content-Type: application/json" \
  -d @- <<'JSON'
{
  "instruction": "You are an AI assistant for beauty-industry information extraction from social media text (TikTok, Instagram, YouTube, Snapchat), including hashtags and @brand mentions.\n\nYour task:\n1. Determine whether the content is beauty-related (skincare, makeup, perfume, haircare).\n2. If non-beauty, set \"is_beauty\" to false and return an empty \"relationships\" list.\n3. If beauty-related, set \"is_beauty\" to true and extract brand mentions into \"relationships\".\n\nOutput schema:\n{\n  \"is_beauty\": true/false,\n  \"reasoning\": \"short reason\",\n  \"relationships\": [\n    {\n      \"brand_text\": \"original matched text\",\n      \"start\": \"start offset\",\n      \"end\": \"end offset\"\n    }\n  ]\n}\n\nRules:\n- Keep only beauty-related brand mentions.\n- For mixed-category brands (e.g., Gucci, Prada), keep only when context is beauty-related.\n- Distinguish person names from brand names by context.\n- Keep offsets aligned with the original text.\n- Return JSON only. No extra explanation outside JSON.",
  "content": "hello \nwe’ve got a neal & wolf facebook live coming up this evening at 6.30pm uk time on salon promotions instagram and we’d love you to join that session. \njoin our head educator laura and senior salon & digital educator chloe live for an essential session on hair health! \nthey’ll be exploring how to identify and treat damage, the difference between moisture and protein, and why hydration is key for pre and post-holiday hair. you’ll get expe advice on building effective haircare routines, using the new hydrate leave-in conditioner, and retailing in-salon with confidence.\nplus, they’ll be doing a live styling demo, featuring tips for caring for curly hair, product layering, and styling techniques for different textures.\nsee you there…..",
  "max_new_tokens": 256
}
JSON

{
  "is_beauty": true,
  "reasoning": "The content is about a Facebook Live session focusing on hair health, haircare routines, and styling demos, featuring the brand Neal & Wolf and their product 'Hydrate Leave-In Conditioner'. This falls under haircare.",
  "relationships": [
    {
      "brand_text": "neal & wolf",
      "start": "19",
      "end": "30"
    }
  ]
}

56
curl --max-time 120 -sS http://127.0.0.1:8012/generate \
  -H "Content-Type: application/json" \
  -d @- <<'JSON'
{
  "instruction": "You are an AI assistant for beauty-industry information extraction from social media text (TikTok, Instagram, YouTube, Snapchat), including hashtags and @brand mentions.\n\nYour task:\n1. Determine whether the content is beauty-related (skincare, makeup, perfume, haircare).\n2. If non-beauty, set \"is_beauty\" to false and return an empty \"relationships\" list.\n3. If beauty-related, set \"is_beauty\" to true and extract brand mentions into \"relationships\".\n\nOutput schema:\n{\n  \"is_beauty\": true/false,\n  \"reasoning\": \"short reason\",\n  \"relationships\": [\n    {\n      \"brand_text\": \"original matched text\",\n      \"start\": \"start offset\",\n      \"end\": \"end offset\"\n    }\n  ]\n}\n\nRules:\n- Keep only beauty-related brand mentions.\n- For mixed-category brands (e.g., Gucci, Prada), keep only when context is beauty-related.\n- Distinguish person names from brand names by context.\n- Keep offsets aligned with the original text.\n- Return JSON only. No extra explanation outside JSON.",
  "content": "this week in fashion: moncler turns up the joy factor with its new campaign, while hermès men’s designer véronique nichanian will exit after 37 years — plus other key industry moves.\n1. dubai-based retail firm @chalhoubgroup said it invested in @willychavarria, with plans to help scale the ready-to-wear brand into new markets including europe and asia.\n2. @moncler’s new campaign starring robe de niro and al pacino struck a warm, nostalgic chord — and earned praise across social media for its sincerity.\n3. @mariagraziachiuri will return to @fendi as chief creative officer, succeeding silvia venturini fendi and returning to a label where she worked for most of the 1990s. chiuri stepped down as aistic director at christian @dior earlier this year.\n4. @lvmh shares jumped this week as its growth rebounded in its latest quaer, fuelled by @sephora’s retail boom and recovery in china, even as fashion and leather goods continue to lag.\n5. amid its stalled turnaround, the @victoriassecret fashion show this year featured big hair and even bigger wings, a nod to its early-aughts peak of glitz and glossy sex appeal.\n6. from “bush thong” to king kylie to @skims beauty, kardashian headlines are coming so fast, they’re staing to cancel each other out.\n7. the eu commission fined @gucci, @chloe and @loewe 157 million ($181.52 million) for anticompetitive price-fixing practices that reduced consumer choice, such as imposition restrictions on retail prices and discount rates.\n8. perfumers are finding that young consumers are no longer wearing signature scents, instead layering up to a handful at once, challenging how hero products like chanel no. 5 and miss dior can be marketed.\n9. @hermès men’s designer véronique nichanian will exit after 37 years, closing an era of refined tailoring before her final show in january.\nthis week in fashion is bof’s weekly news briefing, available exclusively to bof professional members. become a bof professional member to access the full analysis #link in bio \npaul yem (@paulyem), moncler, paola mattioli, getty images, spotlight/launchmetrics.com, kylie jenner @kyliejenner, shutterstock, unsplash, spotlight/launchmetrics.com",
  "max_new_tokens": 256
}
JSON

{
  "is_beauty": false,
  "reasoning": "The content is entirely focused on the fashion industry (Moncler, Hermès, Fendi, Dior, Gucci, Loewe, Victoria's Secret, Skims fashion/brand news) and retail/business news (LVMH, Sephora performance). While Skims Beauty is mentioned, the context is primarily fashion headlines, and no specific beauty products or categories are detailed enough to warrant an 'is_beauty: true' classification based on the rules (which prioritize explicit beauty product features over passing mentions in a fashion roundup). Sephora is a retailer, not a brand.",
  "relationships": []
}

64
curl --max-time 120 -sS http://127.0.0.1:8012/generate \
  -H "Content-Type: application/json" \
  -d @- <<'JSON'
{
  "instruction": "You are an AI assistant for beauty-industry information extraction from social media text (TikTok, Instagram, YouTube, Snapchat), including hashtags and @brand mentions.\n\nYour task:\n1. Determine whether the content is beauty-related (skincare, makeup, perfume, haircare).\n2. If non-beauty, set \"is_beauty\" to false and return an empty \"relationships\" list.\n3. If beauty-related, set \"is_beauty\" to true and extract brand mentions into \"relationships\".\n\nOutput schema:\n{\n  \"is_beauty\": true/false,\n  \"reasoning\": \"short reason\",\n  \"relationships\": [\n    {\n      \"brand_text\": \"original matched text\",\n      \"start\": \"start offset\",\n      \"end\": \"end offset\"\n    }\n  ]\n}\n\nRules:\n- Keep only beauty-related brand mentions.\n- For mixed-category brands (e.g., Gucci, Prada), keep only when context is beauty-related.\n- Distinguish person names from brand names by context.\n- Keep offsets aligned with the original text.\n- Return JSON only. No extra explanation outside JSON.",
  "content": "“hi loves  i’m chloe — a hairstylist for 15 years and a proud mom of 4. this page is all about hair, family, and real life. expect transformations, quick hair hacks for busy moms, and a little behind-the-scenes of juggling salon + mom life ‍#hair #hair transformation #schwarzkopf #highlights #balayageaists  #viral  #clientlove #hair goals #blonde hair #hairstylist life",
  "max_new_tokens": 256
}
JSON

{
  "is_beauty": true,
  "reasoning": "The content is about hairstyling, transformations, and hair hacks, which falls under haircare. The brand Schwarzkopf is mentioned. 'Chloe' is the creator's name, not a brand. Categories include hair treatment/styling.",
  "relationships": [
    {
      "brand_text": "schwarzkopf",
      "start": "269",
      "end": "280"
    }
  ]
}

66
curl --max-time 120 -sS http://127.0.0.1:8012/generate \
  -H "Content-Type: application/json" \
  -d @- <<'JSON'
{
  "instruction": "You are an AI assistant for beauty-industry information extraction from social media text (TikTok, Instagram, YouTube, Snapchat), including hashtags and @brand mentions.\n\nYour task:\n1. Determine whether the content is beauty-related (skincare, makeup, perfume, haircare).\n2. If non-beauty, set \"is_beauty\" to false and return an empty \"relationships\" list.\n3. If beauty-related, set \"is_beauty\" to true and extract brand mentions into \"relationships\".\n\nOutput schema:\n{\n  \"is_beauty\": true/false,\n  \"reasoning\": \"short reason\",\n  \"relationships\": [\n    {\n      \"brand_text\": \"original matched text\",\n      \"start\": \"start offset\",\n      \"end\": \"end offset\"\n    }\n  ]\n}\n\nRules:\n- Keep only beauty-related brand mentions.\n- For mixed-category brands (e.g., Gucci, Prada), keep only when context is beauty-related.\n- Distinguish person names from brand names by context.\n- Keep offsets aligned with the original text.\n- Return JSON only. No extra explanation outside JSON.",
  "content": "she walked in with a wig that felt more like a costume than comfo — awkward, bulky, and anything but empowering. but watch maria’s transformation… as she discovers a look that’s natural, flattering and brings back the glow she thought she lost. \ntoo often, those facing medical hair loss are handed itchy, outdated wigs that only add to the burden. at eva & chloe, we believe you deserve better — because your beauty, dignity, and confidence should never be compromised.\nour wigs are the result of years of innovation, obsessing over every detail to create something that feels *real* — because it is. real craftsmanship. real transformation. real you. \nthank you, maria, for trusting us with your journey.\n#evaandchloe #wigswithhea #wig transformation #wigsforcancer #wigsforalopecia #hair loss help #lacetop #natural wig #customwig #confidencerestored #wigglow up #modern salon #alopecia awareness #evaandchloewig",
  "max_new_tokens": 256
}
JSON

{
  "is_beauty": true,
  "reasoning": "The content discusses wigs for medical hair loss, focusing on the product (wigs) and transformation, which falls under haircare/beauty accessories, despite the initial negative comparison to fashion wigs. 'Eva & Chloe' is the brand providing the wigs. The hashtags confirm the focus on wigs and hair loss solutions.",
  "relationships": [
    {
      "brand_text": "eva & chloe",
      "start": "352",
      "end": "363"
    }
  ]
}

68
curl --max-time 120 -sS http://127.0.0.1:8012/generate \
  -H "Content-Type: application/json" \
  -d @- <<'JSON'
{
  "instruction": "You are an AI assistant for beauty-industry information extraction from social media text (TikTok, Instagram, YouTube, Snapchat), including hashtags and @brand mentions.\n\nYour task:\n1. Determine whether the content is beauty-related (skincare, makeup, perfume, haircare).\n2. If non-beauty, set \"is_beauty\" to false and return an empty \"relationships\" list.\n3. If beauty-related, set \"is_beauty\" to true and extract brand mentions into \"relationships\".\n\nOutput schema:\n{\n  \"is_beauty\": true/false,\n  \"reasoning\": \"short reason\",\n  \"relationships\": [\n    {\n      \"brand_text\": \"original matched text\",\n      \"start\": \"start offset\",\n      \"end\": \"end offset\"\n    }\n  ]\n}\n\nRules:\n- Keep only beauty-related brand mentions.\n- For mixed-category brands (e.g., Gucci, Prada), keep only when context is beauty-related.\n- Distinguish person names from brand names by context.\n- Keep offsets aligned with the original text.\n- Return JSON only. No extra explanation outside JSON.",
  "content": "in a season defined by strong colors and unapologetic attitude, this look speaks for itself.\nفي موسم يتّسم بالألوان القوية والمواقف الجريئة، هذه الإطلالة تعبّر عن نفسها\n@chloe \npublisher & group editor: valia taha @valiataha \nphotography: soraya & rodrigo @soraya__rodrigo \nstylist: sergi padial  @sergipadial @mmgaists \nstylist assistant: edgar lópez @edgarlopezda \nmakeup & hair: maresco @victormaresco for dior beauty @diorbeauty \nmakeup & hair assistant: maa rico",
  "max_new_tokens": 256
}
JSON

{
  "is_beauty": true,
  "reasoning": "The content mentions 'Dior Beauty' explicitly in the context of makeup & hair, confirming it is beauty content. Dior Beauty is the brand. No specific product names (SKUs) are listed, but the category is makeup/hair.",
  "relationships": [
    {
      "brand_text": "dior beauty",
      "start": "409",
      "end": "420"
    }
  ]
}
