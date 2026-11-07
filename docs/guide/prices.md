# Prices and the rig cost

The library carries cited list prices (docs/models/library.md#prices): one unit's price as the maker's own store or
official price list stated it on the date read, with the page's URL. The rig inventory adds them up as `rig.cost`.
Nothing is estimated and no currency is converted.

## Worked example: the UR5e sample

```python
import dexvantage as dv

rig = dv.sample('ur5e')                      # a Wrist D405 and an Overhead D435
rig.metrics(['rig.cost'])['rig.cost']
# {'value': 700.0, 'unit': 'USD', 'status': 'ok', ...}

dv.price('intel-realsense-d405')
# {'amount': 325, 'currency': 'USD', 'unit': 'each', 'kind': 'maker_store',
#  'source': 'https://store.realsenseai.com/buy-realsense-depth-camera-d405.html', 'read': '2026-11-07', ...}

dv.prices('depth').to_frame()                # every depth camera's price, source and date
```

From the command line the same figure is in the rig inventory's JSON:

    python -m dexvantage --sample ur5e --rig-inventory out/rig.json     # cost: {"value": 700.0, "unit": "USD", ...}

## When a part has no price: the dexhand sample

The dexhand sample's palm camera (OmniVision OVM6948) has no price on a maker's page, so the cost is a lower bound:

```python
dv.sample('dexhand').metrics(['rig.cost'])['rig.cost']
# {'value': 574.0, 'bound': 'at_least', 'status': 'ok',
#  'reason': 'At least 574.00 USD: a price is not stated for 1 of the 3 parts (Palm micro camera ...), ...'}
```

A requirement `rig.cost <= 500 USD` fails on it (at least 574 cannot be at most 500); `rig.cost <= 1000 USD` stays
"not evaluated" until the palm camera has a price. To give it one, add a custom library entry (a JSON file in the user
library, docs/models/library.md, Custom devices) with a `price` block naming your quote (the figures below are placeholders for yours, not a price):

```json
"price": {"amount": 120, "currency": "USD", "unit": "each", "kind": "reseller",
          "source": "https://your-distributor.example/quote-123", "read": "2026-11-07", "what": "quote for 10 pcs"}
```

## In the optimiser

`rig.cost` is one of the optimiser's default objectives. It is used only when every device the search can choose
states a price; otherwise it is dropped with a sentence naming the devices without one. On the UR5e and ALOHA samples
(RealSense cameras only) it is used.

## Prices in another currency

Every built-in price is in US dollars. A custom entry in another currency gives the rig inventory a total in that
currency, but `rig.cost` (a USD figure) reads "not stated" for it, and a rig mixing currencies is refused: no exchange
rate is assumed.

Known limits: list prices without tax, shipping or discounts, on the date read; volume prices (NVIDIA modules, 1000+)
are marked `min_quantity`; only the enabled sensors are counted; prices go stale, and the validation check fails a
price read more than two years ago.
