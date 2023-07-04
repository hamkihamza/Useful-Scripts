const axios = require('axios');
const cheerio = require('cheerio');
const fs = require('fs').promises;

const urls = ['PUT LINKS HERE IN STRING FORMAT'];
let jsonfile = {};

const fetchAndParseUrls = async () => {
  for (let siteUrl of urls) {
    try {
      const response = await axios.get(siteUrl);
      const html = response.data;
      const $ = cheerio.load(html);
      const links = [];
      const title = $('title').text(); 

      $('a').each((i, elem) => {
        let link = $(elem).attr('href');
        if (link) {
          links.push(link);
        }
      });

      jsonfile[title] = links;
    } catch (error) {
      console.error(`Could not fetch website content from ${siteUrl}:`, error);
    }
  }
}

const writeToFile = async () => {
  try {
    await fs.writeFile("links.json", JSON.stringify(jsonfile));
    console.log('File written successfully');
  } catch (error) {
    console.log('Error writing file:', error);
  }
}

const run = async () => {
  await fetchAndParseUrls();
  await writeToFile();
}

run();
