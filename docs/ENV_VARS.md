# Optional environment variables

Use these when running the API server, connectors, or scripts. Copy into a `.env` file in the project root (e.g. `python-dotenv` or your shell) if you use them.

## Server

| Variable | Description |
|----------|-------------|
| `API_AUTH_TOKEN` | If set, all `/api/*` routes require `Authorization: Bearer <token>`. |

## Cache

| Variable | Description |
|----------|-------------|
| `REDIS_URL` | Redis URL (e.g. `redis://localhost:6379/0`). If unset, in-memory cache is used. |

## External connectors

Only needed if you use these features.

| Variable | Purpose |
|----------|---------|
| `FRED_API_KEY` | FRED economic data connector |
| `GOOGLE_ADS_*` | Google Ads (customer_id, developer_token, client_id, client_secret, refresh_token) |
| `META_*` | Meta Ads (app_id, app_secret, access_token, ad_account_id) |
| `AMAZON_ADS_*` | Amazon Ads (client_id, client_secret, refresh_token, profile_id) |
| `TIKTOK_*` | TikTok Ads (access_token, advertiser_id) |

## Config path

Config is loaded from `config.yaml` or `config/config.yaml` in the project root. To use a different file, pass it via CLI: `python -m cli run --config /path/to/config.yaml`.
