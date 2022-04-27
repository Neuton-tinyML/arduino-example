#include <string.h>

#include "src/calculator/user_app.h"
#include "src/calculator/protocol.h"
#include "src/calculator/parser.h"
#include "src/calculator/checksum.h"
#include "src/calculator/neuton/neuton.h"


#define LED_RED   52
#define LED_GREEN 53


static bool initialised = 0;


static int channel_make_packet(PacketHeader* hdr, void* payload, size_t size, PacketType type, ErrorCode err)
{
	if (!hdr || ((size + sizeof(PacketHeader) + sizeof(uint16_t)) > (parser_buffer_size())))
		return 1;

	if (size && !payload)
		return 2;

	hdr->preamble = PREAMBLE;
	hdr->type = ANS(type);
	hdr->error = err;
	hdr->size = sizeof(PacketHeader) + size + sizeof(uint16_t);

	if (size)
		memcpy(hdr + 1, payload, size);

	uint16_t crc = crc16_table((uint8_t*) hdr, hdr->size - sizeof(uint16_t), 0);
	memcpy((uint8_t*) hdr + hdr->size - sizeof(uint16_t), &crc, sizeof(uint16_t));

	return 0;
}


static inline void channel_send_packet(void* data, size_t size)
{
	Serial.write((const char*) data, size);
}


static void channel_on_valid_packet(void* data, uint32_t)
{
	PacketHeader* hdr = (PacketHeader*) data;

	if (IS_ANS(hdr->type))
		return;

	hdr->size -= sizeof(PacketHeader) + sizeof(uint16_t);

	switch (PACKET_TYPE(hdr->type))
	{
	case TYPE_MODEL_INFO:
	{
		ModelInfo info = app_model_info();

		if (0 == channel_make_packet(hdr, &info, sizeof(info), TYPE_MODEL_INFO, ERROR_SUCCESS))
			channel_send_packet(data, hdr->size);

		app_reset();

		break;
	}

	case TYPE_DATASET_INFO:
	{
		if (hdr->size < sizeof(DatasetInfo))
			break;

		DatasetInfo* info = (DatasetInfo*) (hdr + 1);

		if (0 == app_dataset_info(info))
		{
			if (0 == channel_make_packet(hdr, NULL, 0, TYPE_DATASET_INFO, ERROR_SUCCESS))
				channel_send_packet(data, hdr->size);
		}
		else
		{
			if (0 == channel_make_packet(hdr, NULL, 0, TYPE_ERROR, ERROR_INVALID_SIZE))
				channel_send_packet(data, hdr->size);
		}

		app_reset();

		break;
	}

	case TYPE_DATASET_SAMPLE:
	{
		float* sample = (float*) (hdr + 1);

		uint32_t sz = 0;
		float* result = app_run_inference(sample, hdr->size, &sz);
		if (result && sz)
		{
			if (0 == channel_make_packet(hdr, result, sz, TYPE_DATASET_SAMPLE, ERROR_SUCCESS))
				channel_send_packet(data, hdr->size);
		}
		else
		{
			if (0 == channel_make_packet(hdr, NULL, 0, TYPE_ERROR, result ? ERROR_INVALID_SIZE : ERROR_NO_MEMORY))
				channel_send_packet(data, hdr->size);
		}

		break;
	}

	case TYPE_PERF_REPORT:
	{
		PerformanceReport report;

		report.ramUsage = neuton_model_ram_usage();
		report.ramUsageCur = neuton_model_ram_usage();
		report.bufferSize = parser_buffer_size();
		report.flashUsage = app_model_size();
		report.freq = F_CPU;

		CalculatorStats stats = app_get_stats();

		report.usSampleAvg = stats.usSampleAvg;
		report.usSampleMin = stats.usSampleMin;
		report.usSampleMax = stats.usSampleMax;

		if (0 == channel_make_packet(hdr, &report, sizeof(report), TYPE_PERF_REPORT, ERROR_SUCCESS))
			channel_send_packet(data, hdr->size);

		break;
	}

	case TYPE_ERROR:
	default:
		break;
	}
}


void channel_init(uint32_t size)
{
	initialised = (0 == parser_init(channel_on_valid_packet, size));
}

static void led(uint8_t pin, uint8_t state)
{
	digitalWrite(pin, state ? HIGH : LOW);
}

extern "C" void led_red(uint8_t state)
{
  led(LED_RED, state);
}

extern "C" void led_green(uint8_t state)
{
  led(LED_GREEN, state);
}

void setup()
{
	pinMode(LED_RED, OUTPUT);
	pinMode(LED_GREEN, OUTPUT);

	led_red(1);
	led_green(1);

	initialised = (0 == app_init());
	initialised &= (0 == parser_init(channel_on_valid_packet, app_inputs_size()));

	Serial.begin(230400);
}

void loop()
{
	if (!initialised)
	{
		while(1)
		{
			led_red(1);
			led_green(0);
			delay(100);
			led_red(0);
			led_green(1);
			delay(100);
		}
	}

	while (Serial.available() > 0)
		parser_parse(Serial.read());
}
