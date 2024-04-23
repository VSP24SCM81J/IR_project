# webcrawler/spiders/my_spider.py

import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor

class MySpider(CrawlSpider):
    name = 'webdocumentcrawler'
    allowed_domains = ['naxussolution.com']  # Update this to your domain
    start_urls = ['http://www.naxussolution.com']  # Update this with your starting URL
    custom_settings = {
        'DEPTH_LIMIT': 3,  # Max depth, can be set as an input parameter
        'CLOSESPIDER_PAGECOUNT': 100,  # Max pages, can be set as an input parameter
        'AUTOTHROTTLE_ENABLED': True,  # Enable AutoThrottle
        'DOWNLOAD_DELAY': 1  # Adjust based on the domain's tolerance
    }

    rules = (
        Rule(LinkExtractor(), callback='parse_item', follow=True),
    )

    def parse_item(self, response):
        yield {
            'url': response.url,
            'html': response.text
        }
