
import csv
import scrapy
from scrapy import signals
from scrapy.crawler import CrawlerProcess
from scrapy.signalmanager import dispatcher

class FreshNewSpider(scrapy.Spider):
    name = 'books'

    def start_requests(self):
        # URL = 'https://en.wikipedia.org/wiki/Apocalypse_of_Peter'
        URL = 'https://en.wikipedia.org/wiki/Artificial_intelligence'
        yield scrapy.Request(url=URL, callback=self.response_parser)

    def response_parser(self, response):
        for selector in response.css('.mw-parser-output'):  # Adjust selector to target the correct elements
            paragraphs = selector.css('p').xpath('string(.)').getall() # Get all text content from <p> elements
            # paragraphs = selector.xpath('//p[not(descendant::sup)]/text()').getall()

            yield {
                'title': ' '.join(paragraphs)  # Combine all paragraph texts into a single string
            }

        # next_page_link = response.css('li.next a::attr(href)').extract_first()
        # if next_page_link:
        #     yield response.follow(next_page_link, callback=self.response_parser)



def book_spider_result():
    books_results = []

    def crawler_results(item):
        books_results.append(item)

    dispatcher.connect(crawler_results, signal=signals.item_scraped)
    crawler_process = CrawlerProcess()
    crawler_process.crawl(FreshNewSpider)
    crawler_process.start()
    return books_results


if __name__ == '__main__':
    article_data = book_spider_result()
    text = ""
    for d in article_data:
        text += d['title'] + "\n"
    print("article_data: ", article_data)
    # keys = books_data[0].keys()
    # print(keys
    #       )
    with open('data/wikipedia01.txt', 'w', newline='') as output_file_name:
        output_file_name.write(text)
        output_file_name.close()

