// Unit tests for the scene player behind the project-page animations.
// Run with:  node --test tests/
const test = require('node:test');
const assert = require('node:assert/strict');
const { createPlayer } = require('../static/js/anim.js');

// A manual clock so scene timing is tested deterministically.
function fakeTimer() {
  let now = 0;
  let nextId = 0;
  const queue = new Map();
  return {
    set(fn, ms) {
      nextId += 1;
      queue.set(nextId, { at: now + ms, fn });
      return nextId;
    },
    clear(id) {
      queue.delete(id);
    },
    advance(ms) {
      const target = now + ms;
      for (;;) {
        let due = null;
        for (const [id, entry] of queue) {
          if (entry.at <= target && (due === null || entry.at < due[1].at)) due = [id, entry];
        }
        if (due === null) break;
        queue.delete(due[0]);
        now = due[1].at;
        due[1].fn();
      }
      now = target;
    },
    pending() {
      return queue.size;
    },
  };
}

function setup(opts = {}) {
  const timer = fakeTimer();
  const seen = [];
  const player = createPlayer({
    durations: [1000, 2000, 1500],
    onScene: (i) => seen.push(i),
    timer,
    ...opts,
  });
  return { timer, seen, player };
}

test('play shows scene 0 and advances after its duration', () => {
  const { timer, seen, player } = setup();
  player.play();
  assert.deepEqual(seen, [0]);
  timer.advance(999);
  assert.deepEqual(seen, [0]);
  timer.advance(1);
  assert.deepEqual(seen, [0, 1]);
  assert.equal(player.scene, 1);
});

test('loops back to scene 0 after the last scene', () => {
  const { timer, seen, player } = setup();
  player.play();
  timer.advance(1000 + 2000 + 1500);
  assert.deepEqual(seen, [0, 1, 2, 0]);
});

test('pause stops advancing and play resumes the current scene', () => {
  const { timer, seen, player } = setup();
  player.play();
  timer.advance(1000);
  player.pause();
  timer.advance(10000);
  assert.deepEqual(seen, [0, 1]);
  assert.equal(timer.pending(), 0);
  player.play();
  assert.equal(player.scene, 1);
  timer.advance(2000);
  assert.deepEqual(seen, [0, 1, 1, 2]);
});

test('go jumps to a scene and keeps advancing from it', () => {
  const { timer, seen, player } = setup();
  player.play();
  player.go(2);
  assert.deepEqual(seen, [0, 2]);
  timer.advance(1500);
  assert.deepEqual(seen, [0, 2, 0]);
  assert.equal(timer.pending(), 1);
});

test('replay restarts from scene 0', () => {
  const { timer, seen, player } = setup();
  player.play();
  timer.advance(1000);
  player.replay();
  assert.deepEqual(seen, [0, 1, 0]);
  timer.advance(1000);
  assert.deepEqual(seen, [0, 1, 0, 1]);
});

test('reduced motion shows the final scene and never advances', () => {
  const { timer, seen, player } = setup({ reduced: true });
  player.play();
  assert.deepEqual(seen, [2]);
  timer.advance(100000);
  assert.deepEqual(seen, [2]);
  assert.equal(timer.pending(), 0);
});

test('reduced motion still allows manual stepping', () => {
  const { timer, seen, player } = setup({ reduced: true });
  player.play();
  player.go(1);
  assert.deepEqual(seen, [2, 1]);
  timer.advance(100000);
  assert.deepEqual(seen, [2, 1]);
});
