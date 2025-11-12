.PHONY: lint test setup-configs

NATS_VERSION ?= v2.10.28
ETCD_VERSION ?= v3.5.21

default:
	./run_dashboard.sh

lint:
	uvx pre-commit run --all-files
	uvx ty check

test:
	cd /Users/idhanani/Desktop/benchmarks/infbench && uv run python -m tests.test_basic && uv run python -m tests.test_aggregations

setup:
	@echo "📦 Setting up configs directory..."
	@mkdir -p configs
	@cp deepep_config.json configs/
	@ARCH=$$(uname -m); \
	case "$$ARCH" in \
		x86_64)  ARCH_SHORT="amd64" ;; \
		aarch64) ARCH_SHORT="arm64" ;; \
		*) echo "❌ Unsupported architecture: $$ARCH"; exit 1 ;; \
	esac; \
	echo "⬇️  Downloading Python wheels..."; \
	wget -q --show-progress -P configs https://files.pythonhosted.org/packages/dc/b7/62fb0edaeae0943731d0e1d3e1455b0a8a94ef448aa5bd8ffe33288ab464/ai_dynamo-0.6.1-py3-none-any.whl; \
	wget -q --show-progress -P configs https://files.pythonhosted.org/packages/b8/0c/076268db6ff2c87663a0d70f7ce7a6a1c566ac1383981d9c82437de2ff98/ai_dynamo_runtime-0.6.1-cp310-abi3-manylinux_2_28_aarch64.whl; \
	echo "⬇️  Downloading NATS ($(NATS_VERSION)) for $$ARCH_SHORT..."; \
	NATS_DEB="nats-server-$(NATS_VERSION)-$$ARCH_SHORT.deb"; \
	NATS_URL="https://github.com/nats-io/nats-server/releases/download/$(NATS_VERSION)/$$NATS_DEB"; \
	wget -q --show-progress --tries=3 --waitretry=5 "$$NATS_URL" -O "configs/$$NATS_DEB"; \
	echo "📁 Extracting NATS binary..."; \
	TMP_DIR=$$(mktemp -d); \
	dpkg-deb -x "configs/$$NATS_DEB" "$$TMP_DIR"; \
	if [ -f "$$TMP_DIR/usr/local/bin/nats-server" ]; then \
		cp "$$TMP_DIR/usr/local/bin/nats-server" configs/; \
	elif [ -f "$$TMP_DIR/usr/bin/nats-server" ]; then \
		cp "$$TMP_DIR/usr/bin/nats-server" configs/; \
	else \
		echo "❌ Could not find nats-server binary inside the .deb package"; \
		ls -R "$$TMP_DIR" | head -n 50; \
		exit 1; \
	fi; \
	chmod +x configs/nats-server; \
	rm -rf "$$TMP_DIR" "configs/$$NATS_DEB"; \
	echo "⬇️  Downloading ETCD ($(ETCD_VERSION)) for $$ARCH_SHORT..."; \
	ETCD_TAR="etcd-$(ETCD_VERSION)-linux-$$ARCH_SHORT.tar.gz"; \
	ETCD_URL="https://github.com/etcd-io/etcd/releases/download/$(ETCD_VERSION)/$$ETCD_TAR"; \
	wget -q --show-progress --tries=3 --waitretry=5 "$$ETCD_URL" -O "configs/$$ETCD_TAR"; \
	echo "📁 Extracting ETCD binaries..."; \
	tar -xzf "configs/$$ETCD_TAR" --strip-components=1 -C configs etcd-$(ETCD_VERSION)-linux-$$ARCH_SHORT/etcd etcd-$(ETCD_VERSION)-linux-$$ARCH_SHORT/etcdctl; \
	chmod +x configs/etcd configs/etcdctl; \
	rm "configs/$$ETCD_TAR"; \
	echo "✅ Done. Contents of configs directory:"; \
	ls -lh configs/