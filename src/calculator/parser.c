#include <stddef.h>
#include <string.h>

#include "parser.h"
#include "checksum.h"
#include "protocol.h"


typedef enum
{
	PREAMBLE_1 = 0,
	PREAMBLE_2,
	SIZE,
	TYPE,
	ERROR_CODE,
	RESERVED_BYTES,
	PAYLOAD
}
State;

typedef struct
{
	State           state;
	uint32_t        counter;
	uint32_t        pos;
	uint8_t*        buffer;
	valid_packet_cb callback;
	uint32_t        bufferSize;
}
Parser;

static Parser parser = { 0 };


uint32_t parser_buffer_size()
{
	return parser.bufferSize;
}


uint8_t parser_init(valid_packet_cb callback, uint32_t size)
{
	const uint32_t reportSize =  sizeof(PacketHeader) + sizeof(uint16_t) + sizeof(PerformanceReport);
	parser.bufferSize = sizeof(PacketHeader) + sizeof(uint16_t) + sizeof(float) * size;

	if (parser.bufferSize < reportSize)
		parser.bufferSize = reportSize;

	parser.buffer = malloc(parser.bufferSize);
	parser.callback = callback;

	return parser.buffer != NULL ? 0 : 1;
}


void parser_parse(uint8_t data)
{
	Parser* p = &parser;
	PacketHeader* hdr = (PacketHeader*) parser.buffer;

	switch (p->state)
	{
	case PREAMBLE_1:
		if (data == ((PREAMBLE >> 0) & 0xff))
		{
			p->state = PREAMBLE_2;
			p->pos = 0;
			p->counter = 0;
			p->buffer[p->pos++] = data;
		}
		break;

	case PREAMBLE_2:
		if (data == ((PREAMBLE >> 8) & 0xff))
		{
			p->buffer[p->pos++] = data;
			p->state = hdr->preamble == PREAMBLE ? SIZE : PREAMBLE_1;
		}
		else
			p->state = PREAMBLE_1;
		break;

	case SIZE:
		p->buffer[p->pos++] = data;
		if (++p->counter >= sizeof(hdr->size))
		{
			p->state = (hdr->size <= p->bufferSize) &&
					(hdr->size >= (sizeof(PacketHeader) + sizeof(uint16_t)))
					? TYPE : PREAMBLE_1;
		}
		break;

	case TYPE:
		p->buffer[p->pos++] = data;
		p->state = ERROR_CODE;
		break;

	case ERROR_CODE:
		p->buffer[p->pos++] = data;
		p->state = RESERVED_BYTES;
		p->counter = sizeof(hdr->reserved);
		break;

	case RESERVED_BYTES:
		p->buffer[p->pos++] = data;
		if (--p->counter <= 0)
		{
			p->counter = hdr->size - sizeof(PacketHeader);
			p->state = p->counter >= sizeof(uint16_t) ? PAYLOAD : PREAMBLE_1;
		}
		break;

	case PAYLOAD:
		p->buffer[p->pos++] = data;
		if (--p->counter <= 0)
		{
			p->state = PREAMBLE_1;
			p->pos -= sizeof(uint16_t);

			uint16_t crc_packet;
			memcpy(&crc_packet, &p->buffer[p->pos], sizeof(uint16_t));

			if (crc_packet == crc16_table(p->buffer, hdr->size - sizeof(uint16_t), 0))
			{
				p->callback(p->buffer, hdr->size);
			}
		}
		break;

	default:
		break;
	}
}


void parser_reset()
{
	parser.state = PREAMBLE_1;
}
