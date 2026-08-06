#include "../src/position.h"
#include <iostream>

struct Case {
    const char* name;
    const char* fen;
    int expected;
};

int main() {
    initTables();
    initZobrist();
    initCastlePerm();

    const Case cases[] = {
        {"pawn trade", "4k3/8/2p5/3p4/4P3/8/8/4K3 w - - 0 1", 0},
        {"queen loses to defender", "4k3/8/2p5/3p4/4Q3/8/8/4K3 w - - 0 1", -800},
        {"pawn wins defended queen", "4k3/8/2p5/3q4/4P3/8/8/4K3 w - - 0 1", 800},
    };

    bool ok = true;
    for (const Case& test : cases) {
        Position pos;
        pos.setFen(test.fen);
        const int actual = see(pos, makeMove(28, 35)); // e4xd5
        std::cout << test.name << ": " << actual << " (expected "
                  << test.expected << ")\n";
        ok &= actual == test.expected;
    }
    return ok ? 0 : 1;
}
