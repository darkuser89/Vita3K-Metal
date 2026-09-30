// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <renderer/metal/capabilities.h>
#include <QStringList>

namespace gui {
inline QStringList metal_screen_filters() {
    QStringList filters{QStringLiteral("Nearest"), QStringLiteral("Bilinear")};
    if (renderer::metal::supports_metalfx_spatial())
        filters.append(QStringLiteral("MetalFX Spatial"));
    return filters;
}
}
