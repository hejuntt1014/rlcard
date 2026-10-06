import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as ort from 'onnxruntime-node';
import { InferenceBatcher } from './inferenceBatcher.js';
import { positiveInteger, seededRandom } from './runtime.js';
import { __test__ } from './domain/game/engines/rules/RuleRLBrain.js';
import { Suit, Rank } from '@a3/shared';

test('seed and CPU settings are deterministic and bounded', () => {
  const a=seededRandom(42),b=seededRandom(42);
  assert.deepEqual(Array.from({length:30},a),Array.from({length:30},b));
  assert.equal(positiveInteger(undefined,2,'workers'),2);
  for(const value of ['0','-1','nan','1.5']) assert.throws(()=>positiveInteger(value,2,'workers'));
});

test('inference batches route every candidate to its own state', async () => {
  let calls=0,active=0,peak=0;
  const fake={run:async (feeds: Record<string,ort.Tensor>)=>{
    calls++;active++;peak=Math.max(peak,active);
    await new Promise(resolve=>setTimeout(resolve,5));
    const ids=feeds.state_indices!.data as BigInt64Array,obs=feeds.obs_static!.data as Float32Array;
    const actions=feeds.action_feat!.data as Float32Array;
    const q=Float32Array.from(ids, (id,i)=>obs[Number(id)*556]!+actions[i*111]!);
    active--;
    return {play_q:{data:q},aux_logits:{data:new Float32Array(Number(feeds.obs_static!.dims[0])*17)},
      declare_q:{data:new Float32Array(Number(feeds.obs_static!.dims[0])*2)}};
  }} as unknown as ort.InferenceSession;
  const batcher=new InferenceBatcher(async()=>fake);
  const jobs=Array.from({length:12},(_,i)=>{
    const obs=new Float32Array(556);obs[0]=i*10;
    const actions=new Float32Array(3*111);actions[111]=1;actions[222]=2;
    return batcher.infer(obs,new Float32Array(24*88),actions).then(r=>assert.deepEqual([...r.q],[i*10,i*10+1,i*10+2]));
  });
  await Promise.all(jobs);assert.equal(calls,1);assert.equal(peak,1);
});

test('failed inference rejects every pending game', async()=>{
  const batcher=new InferenceBatcher(async()=>({run:async()=>{throw new Error('fixture failure');}} as unknown as ort.InferenceSession));
  const outcomes=await Promise.allSettled(Array.from({length:3},()=>batcher.infer(new Float32Array(556),new Float32Array(2112),new Float32Array(111))));
  assert.ok(outcomes.every(o=>o.status==='rejected'));
});

test('compact afterstates match the native full-house contract',()=>{
  const cards=(rank:Rank)=>[Suit.Diamond,Suit.Club,Suit.Heart].map(suit=>({rank,suit}));
  const ctx={} as any;
  assert.equal(__test__.computeAfterstate([...cards(Rank.Four),{rank:Rank.Five,suit:Suit.Spade}] as any,ctx).hasThreePairOrFourOne,false);
  assert.equal(__test__.computeAfterstate([...cards(Rank.Four),...cards(Rank.Five)] as any,ctx).hasThreePairOrFourOne,true);
});
