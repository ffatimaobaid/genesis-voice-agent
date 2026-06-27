import asyncio
import json
import re
from playwright.async_api import async_playwright

BASE_URL = "https://genesis-cpo.netlify.app/en/inventory"

JUNK = {
    "Genesis Certified", "Browse Inventory", "Compare", "Register Your Interest",
    "Shopping", "Book a Test Drive", "Offers", "Request a Quote", "Find a Dealer",
    "Genesis", "The Brand", "Motor Show", "Magma", "Support", "Download a Catalog",
    "Contact Us", "Owners", "Change Your Region", "The Brand Overview", "Brand News",
    "One of One", "Motor Show Overview", "Auto Shanghai", "Digital Motor Show",
    "New York Auto Show", "Seoul Motor Show", "Busan Motor Show", "Program Overview",
    "Genesis Magma Racing", "FIA World Endurance Championship", "Show all 30 features",
    "REGISTER YOUR INTEREST", "CAN'T FIND YOUR IDEAL GENESIS?",
}

def parse_detail_text(body_text, href):
    def extract(pattern, text, default="Unknown"):
        m = re.search(pattern, text, re.IGNORECASE)
        return m.group(1).strip() if m else default

    year_match = re.search(r'-(202\d)-', href)
    year = year_match.group(1) if year_match else "Unknown"

    drivetrain = "AWD" if "awd" in href.lower() else "RWD"

    model = "Unknown"
    for m in ["G80", "G90", "GV80"]:
        if f"/{m.lower()}-" in href.lower() or f"/{m.lower()}/" in href.lower():
            model = m
            break

    # Trim from engine field or URL slug
    engine = extract(r'Engine\n([^\n]+)', body_text)
    
    # Trim: everything after model in the page title line
    title_match = re.search(r'(G80|G90|GV80)\s+(.+?)\n', body_text, re.IGNORECASE)
    trim = title_match.group(2).strip() if title_match else engine

    exterior = extract(r'Exterior Color\s*([^\n]+)', body_text)
    interior = extract(r'Interior Color\s*([^\n]+)', body_text)
    fuel_type = extract(r'Fuel Type\s*([^\n]+)', body_text)
    body_type = extract(r'Body Type\s*([^\n]+)', body_text)

    transmission_match = re.search(r'Transmission\s*(Automatic|AUTO|Manual)', body_text, re.IGNORECASE)
    transmission = "Automatic" if transmission_match else "Automatic"

    mileage_match = re.search(r'Mileage\s*([\d,]+)\s*km', body_text, re.IGNORECASE)
    mileage = mileage_match.group(1).replace(",", "") if mileage_match else "Unknown"
    availability = "Available" if mileage != "Unknown" else "Not available"

    price_match = re.search(r'TOTAL PURCHASE PRICE\*?\s*([\d,]+)', body_text)
    price = price_match.group(1).replace(",", "") if price_match else "Unknown"

    certified = "Yes" if "GENESIS CERTIFIED" in body_text else "No"

    return {
        "year": year,
        "model": model,
        "trim": trim,
        "drivetrain": drivetrain,
        "exterior_color": exterior,
        "interior_color": interior,
        "fuel_type": fuel_type,
        "engine": engine,
        "transmission": transmission,
        "body_type": body_type,
        "mileage_km": mileage,
        "availability": availability,
        "price_sar": price,
        "genesis_certified": certified,
        "url": f"https://genesis-cpo.netlify.app{href}",
    }


async def get_all_listing_hrefs(page, model):
    url = f"{BASE_URL}?model={model}"
    await page.goto(url, wait_until="networkidle")
    await page.wait_for_timeout(2000)

    all_hrefs = set()
    page_num = 1

    while True:
        print(f"  Page {page_num}...")
        listings = await page.query_selector_all("a[href*='/en/inventory/']")
        for listing in listings:
            href = await listing.get_attribute("href")
            if href and href != "/en/inventory" and "/en/inventory/" in href:
                if "?" not in href and "#" not in href:
                    all_hrefs.add(href)

        print(f"  {len(all_hrefs)} unique hrefs so far")

        body_text = await page.inner_text("body")
        total_match = re.search(r'Showing\s+\d+\s*-\s*\d+\s+of\s+(\d+)', body_text)
        if total_match:
            total = int(total_match.group(1))
            if len(all_hrefs) >= total:
                print(f"  Got all {total}!")
                break

        clicked = False
        try:
            candidates = await page.query_selector_all("button, a")
            for el in candidates:
                aria = await el.get_attribute("aria-label") or ""
                title = await el.get_attribute("title") or ""
                txt = (await el.inner_text()).strip()
                if any(x in (aria + title + txt).lower() for x in ["next", "›", ">"]):
                    disabled = await el.get_attribute("disabled")
                    aria_disabled = await el.get_attribute("aria-disabled")
                    if disabled or aria_disabled == "true":
                        break
                    await el.scroll_into_view_if_needed()
                    await el.click()
                    await page.wait_for_timeout(2500)
                    page_num += 1
                    clicked = True
                    break
        except Exception as e:
            print(f"  Nav error: {e}")

        if not clicked:
            print(f"  No more pages found")
            break

    return list(all_hrefs)


async def scrape():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        all_cars = []
        models = ["G80", "G90", "GV80"]

        for model in models:
            print(f"\nScraping {model}...")
            hrefs = await get_all_listing_hrefs(page, model)
            print(f"  Total hrefs found: {len(hrefs)}")

            for i, href in enumerate(hrefs):
                print(f"  Detail {i+1}/{len(hrefs)}: ...{href[-45:]}")
                try:
                    await page.goto(f"https://genesis-cpo.netlify.app{href}", wait_until="networkidle")
                    await page.wait_for_timeout(1500)
                    body_text = await page.inner_text("body")

                    car = parse_detail_text(body_text, href)

                    # Features
                    # Features — only grab li items that are actual car features
                    features = []
                    feature_els = await page.query_selector_all("ul li")
                    for el in feature_els:
                        txt = (await el.inner_text()).strip()
                        # Skip if contains newlines (navbar junk spans multiple lines)
                        if "\n" in txt or len(txt.split()) > 8:
                            continue
                        if txt and 5 < len(txt) < 80 and txt not in JUNK:
                            features.append(txt)
                    car["features"] = features

                    all_cars.append(car)

                except Exception as e:
                    print(f"  Error: {e}")

        await browser.close()

        # Deduplicate by URL
        seen = set()
        unique = []
        for car in all_cars:
            if car["url"] not in seen:
                seen.add(car["url"])
                unique.append(car)

        with open("inventory.json", "w") as f:
            json.dump(unique, f, indent=2)

        print(f"\nDone! {len(unique)} listings saved to inventory.json")
        
        # Print sample to verify
        if unique:
            print("\nSample entry:")
            sample = unique[0]
            for k, v in sample.items():
                if k != "features":
                    print(f"  {k}: {v}")
            print(f"  features ({len(sample['features'])}): {sample['features'][:3]}")


asyncio.run(scrape())