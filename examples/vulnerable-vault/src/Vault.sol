// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IERC20 {
    function transfer(address to, uint256 amount) external returns (bool);
}

/// @title Vault
/// @notice INTENTIONALLY VULNERABLE example target for kai-security demos.
///         Do NOT deploy. The bugs below are planted so `kai audit` has
///         something real to find on a tiny, self-contained codebase.
contract Vault {
    mapping(address => uint256) public balances;

    function deposit() external payable {
        balances[msg.sender] += msg.value;
    }

    /// BUG 1 — reentrancy: the caller's balance is zeroed *after* the external
    /// call, with no reentrancy guard. A malicious receiver can re-enter
    /// withdraw() from its fallback and drain the contract, because the balance
    /// is still non-zero on each re-entry. (Zeroing with `= 0` rather than a
    /// checked `-=` is what makes this genuinely exploitable under Solidity
    /// 0.8.x — a checked subtraction would underflow and revert the drain.)
    function withdraw() external {
        uint256 amount = balances[msg.sender];
        require(amount > 0, "nothing to withdraw");
        (bool ok, ) = msg.sender.call{value: amount}("");
        require(ok, "transfer failed");
        balances[msg.sender] = 0;
    }

    /// BUG 2 — unchecked return value: ERC-20 transfer() can return false
    /// instead of reverting; ignoring it lets a failed transfer look like a
    /// success.
    function sweepToken(IERC20 token, address to, uint256 amount) external {
        token.transfer(to, amount);
    }
}
